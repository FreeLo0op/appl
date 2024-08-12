# Copyright (c) 2021 Wenet Community. (authors: Binbin Zhang)
#               2023 Wenet Community. (authors: Dinghao Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Optional
import processor
from datapipes import (WenetRawDatasetSource,
                                     WenetTarShardDatasetSource)
from base_tokenizer import BaseTokenizer
from file_utils import read_symbol_table
import torch
# 定义缓存机制，最多缓存4条音频
from collections import OrderedDict
import torchaudio
from torch.nn.utils.rnn import pad_sequence

class AudioCache:
    def __init__(self, max_size=4):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, path):
        if path in self.cache:
            # 将最近访问的音频移到末尾（表示最新）
            self.cache.move_to_end(path)
            return self.cache[path]
        else:
            return None

    def add(self, path, waveform, sample_rate):
        if len(self.cache) >= self.max_size:
            # 删除最早添加的音频
            self.cache.popitem(last=False)
        self.cache[path] = (waveform, sample_rate)

def load_audio_segment(cache, path, start, end):
    audio_data = cache.get(path)
    if not audio_data:
        # 音频不在缓存中，加载整个音频并加入缓存
        waveform, sample_rate = torchaudio.load(path)
        cache.add(path, waveform, sample_rate)
    else:
        waveform, sample_rate = audio_data

    # 根据start和end切割音频段
    start_frame = int(float(start) * sample_rate)
    end_frame = int(float(end) * sample_rate)
    segment = waveform[:, start_frame:end_frame]

    return segment, sample_rate

def LongAudioDataset(data_type,
            data_list_file,
            tokenizer: Optional[BaseTokenizer] = None,
            conf=None,
            partition=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            tokenizer (BaseTokenizer or None): tokenizer to tokenize
            partition(bool): whether to do data partition in terms of rank
    """
    assert conf is not None
    assert data_type in ['raw', 'shard']

    if data_type == 'raw':
        dataset = WenetRawDatasetSource(data_list_file, partition=partition)
        dataset = dataset.map(processor.parse_json)
    else:
        dataset = WenetTarShardDatasetSource(data_list_file,
                                             partition=partition)

    audio_cache = AudioCache(max_size=4)
    #dataset = dataset.map_ignore_error(processor.decode_wav)
    dataset = dataset.map_ignore_error(partial(decode_longwav, cache=audio_cache))
    # for data in dataset:
    #     print(data)

    speaker_conf = conf.get('speaker_conf', None)
    if speaker_conf is not None:
        assert 'speaker_table_path' in speaker_conf
        speaker_table = read_symbol_table(speaker_conf['speaker_table_path'])
        dataset = dataset.map(
            partial(processor.parse_speaker, speaker_dict=speaker_table))

    # if tokenizer is not None:
    #     dataset = dataset.map(partial(processor.tokenize, tokenizer=tokenizer))

    # filter_conf = conf.get('filter_conf', {})
    # dataset = dataset.filter(partial(processor.filter, **filter_conf))

    resample_conf = conf.get('resample_conf', {})
    dataset = dataset.map(partial(processor.resample, **resample_conf))

    speed_perturb = conf.get('speed_perturb', False)
    if speed_perturb:
        dataset = dataset.map(partial(processor.speed_perturb))
    feats_type = conf.get('feats_type', 'fbank')
    assert feats_type in ['fbank', 'mfcc', 'log_mel_spectrogram']

    frontend = conf.get('dataset_frontend', 'other')#sense_voice有自己的前处理流程，所以在这里要分出来
    if feats_type == 'fbank':
        fbank_conf = conf.get('fbank_conf', {})
        if frontend == 'other':
            dataset = dataset.map(partial(processor.compute_fbank, **fbank_conf))
        else:
            cmvn_conf = conf.get('cmvn_conf', {})
            dataset = dataset.map(partial(processor.compute_fbank_sense_voice, **fbank_conf, **cmvn_conf))
    elif feats_type == 'mfcc':
        mfcc_conf = conf.get('mfcc_conf', {})
        dataset = dataset.map(partial(processor.compute_mfcc, **mfcc_conf))
    elif feats_type == 'log_mel_spectrogram':
        log_mel_spectrogram_conf = conf.get('log_mel_spectrogram_conf', {})
        dataset = dataset.map(
            partial(processor.compute_log_mel_spectrogram,
                    **log_mel_spectrogram_conf))
    spec_aug = conf.get('spec_aug', True)
    spec_sub = conf.get('spec_sub', False)
    spec_trim = conf.get('spec_trim', False)
    if spec_aug:
        spec_aug_conf = conf.get('spec_aug_conf', {})
        dataset = dataset.map(partial(processor.spec_aug, **spec_aug_conf))
    if spec_sub:
        spec_sub_conf = conf.get('spec_sub_conf', {})
        dataset = dataset.map(partial(processor.spec_sub, **spec_sub_conf))
    if spec_trim:
        spec_trim_conf = conf.get('spec_trim_conf', {})
        dataset = dataset.map(partial(processor.spec_trim, **spec_trim_conf))

    language_conf = conf.get('language_conf', {"limited_langs": ['zh', 'en']})
    # dataset = dataset.map(partial(processor.detect_language, **language_conf))
    # dataset = dataset.map(processor.detect_task)
    if frontend == 'sense_voice':
        addf_conf = conf.get('addf_conf', {})
        lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        textnorm_dict = {"withitn": 14, "woitn": 15}
        embed = torch.nn.Embedding(
            7 + len(lid_dict) + len(textnorm_dict), conf['fbank_conf']['num_mel_bins']*conf['fbank_conf']['lfr_m']
        )
        if conf['embed_model_path'] != None:
            embed.load_state_dict(torch.load(conf['embed_model_path']))
        for param in embed.parameters():
            param.requires_grad = False
        addf_conf['embed'] = embed
        # for data in dataset:
        #     print(f"2 fbank——speech: {data['feat']}")
        dataset = dataset.map(partial(processor.add_frame, **addf_conf))


    # shuffle = conf.get('shuffle', True)
    # if shuffle:
    #     shuffle_conf = conf.get('shuffle_conf', {})
    #     dataset = dataset.shuffle(buffer_size=shuffle_conf['shuffle_size'])

    # sort = conf.get('sort', True)
    # if sort:
    #     sort_conf = conf.get('sort_conf', {})
    #     dataset = dataset.sort(buffer_size=sort_conf['sort_size'],
    #                            key_func=processor.sort_by_feats)

    batch_conf = conf.get('batch_conf', {})
    batch_type = batch_conf.get('batch_type', 'static')
    if batch_type == 'static':
        assert 'batch_size' in batch_conf
        batch_size = batch_conf.get('batch_size', 16)
        dataset = dataset.batch(batch_size, wrapper_class=padding)
    elif batch_type == 'bucket':
        assert 'bucket_boundaries' in batch_conf
        assert 'bucket_batch_sizes' in batch_conf
        dataset = dataset.bucket_by_sequence_length(
            processor.feats_length_fn,
            batch_conf['bucket_boundaries'],
            batch_conf['bucket_batch_sizes'],
            wrapper_class=padding)
    else:
        max_frames_in_batch = batch_conf.get('max_frames_in_batch', 12000)
        dataset = dataset.dynamic_batch(
            processor.DynamicBatchWindow(max_frames_in_batch),
            wrapper_class=padding,
        )

    return dataset

def decode_longwav(sample,cache=None):
    """ Parse key/wav/txt from json line

        Args:
            sample: str, str is a json line has key/wav/txt

        Returns:
            {key, wav, sample_rate, ...}
    """
    assert 'key' in sample
    assert 'wav' in sample
    assert 'start' in sample
    assert 'end' in sample
    wav_file = sample['wav']
    segment_wav, sr = load_audio_segment(cache, wav_file, sample['start'], sample['end'])
    # if isinstance(wav_file, str):
    #     with open(wav_file, 'rb') as f:
    #         wav_file = f.read()
    # if 'start' in sample:
    #     assert 'end' in sample
    #     sample_rate = torchaudio.info(wav_file).sample_rate
    #     start_frame = int(sample['start'] * sample_rate)
    #     end_frame = int(sample['end'] * sample_rate)
    #     with io.BytesIO(wav_file) as file_obj:
    #         waveform, _ = torchaudio.load(filepath=file_obj,
    #                                       num_frames=end_frame - start_frame,
    #                                       frame_offset=start_frame)
    # else:
    #     with io.BytesIO(wav_file) as file_obj:
    #         waveform, sample_rate = torchaudio.load(file_obj)
    # del wav_file
    del sample['wav']
    sample['wav'] = segment_wav  # overwrite wav
    sample['sample_rate'] = sr
    return sample

def padding(data):
    """ Padding the data into training data

        Args:
            data: List[{key, feat, label}

        Returns:
            Tuple(keys, feats, labels, feats lengths, label lengths)
    """
    sample = data
    assert isinstance(sample, list)
    feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                dtype=torch.int32)
    order = torch.argsort(feats_length, descending=True)
    feats_lengths = torch.tensor([sample[i]['feat'].size(0) for i in order],
                                 dtype=torch.int32)
    sorted_feats = [sample[i]['feat'] for i in order]
    sorted_keys = [sample[i]['key'] for i in order]
    # sorted_labels = [
    #     torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order
    # ]
    #sorted_tokens = [sample[i]['tokens'] for i in order]
    sorted_wavs = [sample[i]['wav'].squeeze(0) for i in order]
    # langs = [sample[i]['lang'] for i in order]
    # tasks = [sample[i]['task'] for i in order]
    # label_lengths = torch.tensor([x.size(0) for x in sorted_labels],
    #                              dtype=torch.int32)
    wav_lengths = torch.tensor([x.size(0) for x in sorted_wavs],
                               dtype=torch.int32)
    padded_feats = pad_sequence(sorted_feats,
                                batch_first=True,
                                padding_value=0)
    # padding_labels = pad_sequence(sorted_labels,
    #                               batch_first=True,
    #                               padding_value=-1)
    padded_wavs = pad_sequence(sorted_wavs, batch_first=True, padding_value=0)

    batch = {
        "keys": sorted_keys,
        "feats": padded_feats,
        "feats_lengths": feats_lengths,
        "pcm": padded_wavs,
        "pcm_length": wav_lengths,
    }
    if 'speaker' in sample[0]:
        speaker = torch.tensor([sample[i]['speaker'] for i in order],
                               dtype=torch.int32)
        batch['speaker'] = speaker
    return batch