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

import io
import json
from subprocess import PIPE, Popen
from urllib.parse import urlparse
from langid.langid import LanguageIdentifier, model
import logging
import librosa
import random

import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
from base_tokenizer import BaseTokenizer
import numpy as np

torchaudio.utils.sox_utils.set_buffer_size(16500)

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

lid = LanguageIdentifier.from_modelstring(model, norm_probs=True)

logging.getLogger('langid').setLevel(logging.INFO)


class UrlOpenError(Exception):

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self.err_msg = msg

    def __str__(self) -> str:
        return self.err_msg


def parse_json(elem):
    line = elem['line']
    obj = json.loads(line)
    obj['file_name'] = elem['file_name']
    return dict(obj)


def parse_url(elem):
    assert 'file_name' in elem
    assert 'line' in elem
    assert isinstance(elem, dict)
    url = elem['line']
    try:
        pr = urlparse(url)
        # local file
        if pr.scheme == '' or pr.scheme == 'file':
            stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
        else:
            cmd = f'wget -q -O - {url}'
            process = Popen(cmd, shell=True, stdout=PIPE)
            elem.update(process=process)
            stream = process.stdout
        elem.update(stream=stream)
        return elem
    except Exception as ex:
        err_msg = 'Failed to open {}'.format(url)
        raise UrlOpenError(err_msg) from ex


def parse_speaker(sample, speaker_dict):
    assert 'speaker' in sample
    speaker = sample['speaker']
    sample['speaker'] = speaker_dict.get(speaker, 0)
    return sample


def detect_language(sample, limited_langs):
    assert 'txt' in sample
    # NOTE(xcsong): Because language classification may not be very accurate
    #   (for example, Chinese being classified as Japanese), our workaround,
    #   given we know for certain that the training data only consists of
    #   Chinese and English, is to limit the classification results to reduce
    #   the impact of misclassification.
    lid.set_languages(limited_langs)
    # i.e., ('zh', 0.9999999909903544)
    sample['lang'] = lid.classify(sample['txt'])[0]
    return sample


def detect_task(sample):
    # TODO(xcsong): Currently, the task is hard-coded to 'transcribe'.
    #   In the future, we could dynamically determine the task based on
    #   the contents of sample. For instance, if a sample contains both
    #   'txt_en' and 'txt_zh', the task should be set to 'translate'.
    sample['task'] = "transcribe"
    return sample


def decode_wav(sample):
    """ Parse key/wav/txt from json line

        Args:
            sample: str, str is a json line has key/wav/txt

        Returns:
            {key, wav, sample_rate, ...}
    """
    assert 'key' in sample
    assert 'wav' in sample
    assert 'txt' in sample
    wav_file = sample['wav']
    if isinstance(wav_file, str):
        with open(wav_file, 'rb') as f:
            wav_file = f.read()
    if 'start' in sample:
        assert 'end' in sample
        sample_rate = torchaudio.load(wav_file).sample_rate
        start_frame = int(sample['start'] * sample_rate)
        end_frame = int(sample['end'] * sample_rate)
        with io.BytesIO(wav_file) as file_obj:
            waveform, _ = torchaudio.load(filepath=file_obj,
                                          num_frames=end_frame - start_frame,
                                          frame_offset=start_frame)
    else:
        with io.BytesIO(wav_file) as file_obj:
            waveform, sample_rate = torchaudio.load(file_obj)
    # del wav_file
    del sample['wav']
    sample['wav'] = waveform  # overwrite wav
    sample['sample_rate'] = sample_rate
    return sample


def resample(sample, resample_rate=16000):
    """ Resample sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            resample_rate: target resample rate

        Returns:
            {key, wav, label, sample_rate}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    if sample_rate != resample_rate:
        sample['sample_rate'] = resample_rate
        sample['wav'] = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    return sample


def speed_perturb(sample, speeds=None):
    """ Apply speed perturb to the sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            speeds(List[float]): optional speed

        Returns:
            key, wav, label, sample_rate}
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    assert 'sample_rate' in sample
    assert 'wav' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    speed = random.choice(speeds)
    if speed != 1.0:
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate,
            [['speed', str(speed)], ['rate', str(sample_rate)]])
        sample['wav'] = wav

    return sample


def compute_fbank(sample,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            sample: {key, wav, sample_rate, ...}

        Returns:
            {key, feat, wav, sample_rate, ...}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    assert 'key' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    waveform = waveform * (1 << 15)
    # Only keep key, feat, label
    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      energy_floor=0.0,
                      sample_frequency=sample_rate)
    sample['feat'] = mat
    return sample


def compute_w2vbert_fbank(sample,
                          num_mel_bins=23,
                          frame_length=25,
                          frame_shift=10,
                          dither=0.0):
    """ Extract Pretrain w2vbert(4.5M hours) fbank
    """
    sample = compute_fbank(sample, num_mel_bins, frame_length, frame_shift,
                           dither)
    mat = sample['feat']
    std, mean = torch.std_mean(mat, dim=0)
    mat = mat.subtract(mean).divide(std)
    sample['feat'] = mat
    return sample


def sort_by_feats(sample):
    assert 'feat' in sample
    assert isinstance(sample['feat'], torch.Tensor)
    return sample['feat'].size(0)


def feats_length_fn(sample) -> int:
    assert 'feat' in sample
    return sample['feat'].size(0)


def compute_mfcc(sample,
                 num_mel_bins=23,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.0,
                 num_ceps=40,
                 high_freq=0.0,
                 low_freq=20.0):
    """ Extract mfcc

        Args:
            sample: {key, wav, sample_rate, ...}

        Returns:
            {key, wav, feat, sample_rate, ...}
    """
    assert 'wav' in sample
    assert 'key' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    waveform = waveform * (1 << 15)
    mat = kaldi.mfcc(waveform,
                     num_mel_bins=num_mel_bins,
                     frame_length=frame_length,
                     frame_shift=frame_shift,
                     dither=dither,
                     num_ceps=num_ceps,
                     high_freq=high_freq,
                     low_freq=low_freq,
                     sample_frequency=sample_rate)
    sample['feat'] = mat
    return sample


def compute_log_mel_spectrogram(sample,
                                n_fft=400,
                                hop_length=160,
                                num_mel_bins=80,
                                padding=0,
                                pad_or_trim: bool = False,
                                max_duration: int = 30):
    """ Extract log mel spectrogram, modified from openai-whisper, see:
        - https://github.com/openai/whisper/blob/main/whisper/audio.py
        - https://github.com/wenet-e2e/wenet/pull/2141#issuecomment-1811765040

        Args:
            sample: {key, wav, sample_rate, ...}
            max_duration: valid when pad_or_trim is True (orign whisper style)

        Returns:
            {key, feat, wav, sample_rate, ...}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    assert 'key' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav'].squeeze(0)  # (channel=1, sample) -> (sample,)
    if padding > 0:
        waveform = F.pad(waveform, (0, padding))
    if pad_or_trim:
        length = max_duration * sample_rate
        if waveform.size(0) >= length:
            waveform = waveform[:length]
        else:
            waveform = F.pad(waveform, (0, length - waveform.size(0)))

    window = torch.hann_window(n_fft)
    stft = torch.stft(waveform,
                      n_fft,
                      hop_length,
                      window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs()**2

    filters = torch.from_numpy(
        librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mel_bins))
    mel_spec = filters @ magnitudes

    # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    sample['feat'] = log_spec.transpose(0, 1)
    return sample


def tokenize(sample, tokenizer: BaseTokenizer):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            sample: {key, wav, txt, sample_rate, ...}

        Returns:
            {key, wav, txt, tokens, label, sample_rate, ...}
    """
    assert 'txt' in sample
    tokens, label = tokenizer.tokenize(sample['txt'])
    sample['tokens'] = tokens
    sample['label'] = label
    return sample


def filter(sample,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            sample: {key, wav, label, sample_rate, ...}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            bool: True to keep, False to filter
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    # sample['wav'] is torch.Tensor, we have 100 frames every second
    num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
    if num_frames < min_length:
        return False
    if num_frames > max_length:
        return False

    if 'label' in sample:
        if len(sample['label']) < token_min_length:
            return False
        if len(sample['label']) > token_max_length:
            return False
        if num_frames != 0:
            if len(sample['label']) / num_frames < min_output_input_ratio:
                return False
            if len(sample['label']) / num_frames > max_output_input_ratio:
                return False
    return True


def spec_aug(sample, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            sample: {key, feat, ...}
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            {key, feat, ....}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    y = x.clone().detach()
    max_frames = y.size(0)
    max_freq = y.size(1)
    # time mask
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0
    # freq mask
    for _ in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    sample['feat'] = y
    return sample


def spec_sub(sample, max_t=20, num_t_sub=3):
    """ Do spec substitute
        Inplace operation
        ref: U2++, section 3.2.3 [https://arxiv.org/abs/2106.05642]

        Args:
            sample: Iterable{key, feat, ...}
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns
            {key, feat, ...}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    y = x.clone().detach()
    max_frames = y.size(0)
    for _ in range(num_t_sub):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        # only substitute the earlier time chosen randomly for current time
        pos = random.randint(0, start)
        y[start:end, :] = x[start - pos:end - pos, :]
    sample['feat'] = y
    return sample


def spec_trim(sample, max_t=20):
    """ Trim tailing frames. Inplace operation.
        ref: TrimTail [https://arxiv.org/abs/2211.00522]

        Args:
            sample: {key, feat, label}
            max_t: max width of length trimming

        Returns:
            {key, feat, label}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    max_frames = x.size(0)
    length = random.randint(1, max_t)
    if length < max_frames / 2:
        y = x.clone().detach()[:max_frames - length]
        sample['feat'] = y
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
    sorted_labels = [
        torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order
    ]
    sorted_tokens = [sample[i]['tokens'] for i in order]
    sorted_wavs = [sample[i]['wav'].squeeze(0) for i in order]
    langs = [sample[i]['lang'] for i in order]
    tasks = [sample[i]['task'] for i in order]
    label_lengths = torch.tensor([x.size(0) for x in sorted_labels],
                                 dtype=torch.int32)
    wav_lengths = torch.tensor([x.size(0) for x in sorted_wavs],
                               dtype=torch.int32)
    padded_feats = pad_sequence(sorted_feats,
                                batch_first=True,
                                padding_value=0)
    padding_labels = pad_sequence(sorted_labels,
                                  batch_first=True,
                                  padding_value=-1)
    padded_wavs = pad_sequence(sorted_wavs, batch_first=True, padding_value=0)

    batch = {
        "keys": sorted_keys,
        "feats": padded_feats,
        "target": padding_labels,
        "feats_lengths": feats_lengths,
        "target_lengths": label_lengths,
        "pcm": padded_wavs,
        "pcm_length": wav_lengths,
        "langs": langs,
        "tasks": tasks,
        "tokens": sorted_tokens,
    }
    if 'speaker' in sample[0]:
        speaker = torch.tensor([sample[i]['speaker'] for i in order],
                               dtype=torch.int32)
        batch['speaker'] = speaker
    return batch

def combine_multi_datapipes(data):
    batch = {}
    max_feat_length = 0
    feats_length_list = []
    keys_list = []
    feats_list = []   #padding
    target_list = []  #padding
    target_length_list = []
    pcm_list = []  #padding
    pcm_length_list = []
    langs_list = []
    tasks_list = []
    for data_i in data:
        max_feat_length = max(max_feat_length, data_i['feats_lengths'][0])
        feats_length_list.append(data_i['feats_lengths'])
        target_length_list.append(data_i['target_lengths'])
        pcm_length_list.append(data_i['pcm_length'])
        keys_list += data_i['keys']
        feats_list += [item.squeeze(0) for item in torch.split(data_i['feats'], 1, dim=0)]
        target_list += [item.squeeze(0) for item in torch.split(data_i['target'], 1, dim=0)]
        pcm_list += [item.squeeze(0) for item in torch.split(data_i['pcm'], 1, dim=0)]
        langs_list += data_i['langs']
        tasks_list += data_i['tasks']

    feats_length = torch.cat(feats_length_list)
    targets_length = torch.cat(target_length_list)
    wav_length = torch.cat(pcm_length_list)
    order = torch.argsort(feats_length, descending=True)
    sorted_keys = [keys_list[i] for i in order]
    langs = [langs_list[i] for i in order]
    tasks = [tasks_list[i] for i in order]
    feats_lengths = feats_length[order]
    label_lengths = targets_length[order]
    wav_lengths = wav_length[order]

    sorted_feats = [feats_list[i] for i in order]
    sorted_target = [target_list[i] for i in order]
    sorted_wavs = [pcm_list[i] for i in order]
    padded_feats = pad_sequence(sorted_feats,
                        batch_first=True,
                        padding_value=0)
    padding_labels = pad_sequence(sorted_target,
                            batch_first=True,
                            padding_value=-1)
    padded_wavs = pad_sequence(sorted_wavs, batch_first=True, padding_value=0)
    
        
    batch = {
    "keys": sorted_keys,
    "feats": padded_feats,
    "target": padding_labels,
    "feats_lengths": feats_lengths,
    "target_lengths": label_lengths,
    "pcm": padded_wavs,
    "pcm_length": wav_lengths,
    "langs": langs,
    "tasks": tasks,
    }
    return batch

class DynamicBatchWindow:

    def __init__(self, max_frames_in_batch=12000):
        self.longest_frames = 0
        self.max_frames_in_batch = max_frames_in_batch

    def __call__(self, sample, buffer_size):
        assert isinstance(sample, dict)
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        new_sample_frames = sample['feat'].size(0)
        self.longest_frames = max(self.longest_frames, new_sample_frames)
        frames_after_padding = self.longest_frames * (buffer_size + 1)
        if frames_after_padding > self.max_frames_in_batch:
            self.longest_frames = new_sample_frames
            return True
        return False


def apply_lfr(inputs, lfr_m, lfr_n):
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, inputs))
    T = T + (lfr_m - 1) // 2
    for i in range(T_lfr):
        if lfr_m <= T - i * lfr_n:
            LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).view(1, -1))
        else:  # process last LFR frame
            num_padding = lfr_m - (T - i * lfr_n)
            frame = (inputs[i * lfr_n :]).view(-1)
            for _ in range(num_padding):
                frame = torch.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    LFR_outputs = torch.vstack(LFR_inputs)
    return LFR_outputs.type(torch.float32)

def apply_cmvn(inputs, cmvn):  # noqa
    """
    Apply CMVN with mvn data
    """

    device = inputs.device
    dtype = inputs.dtype
    frame, dim = inputs.shape

    means = cmvn[0:1, :dim]
    vars = cmvn[1:2, :dim]
    inputs += means.to(device)
    inputs *= vars.to(device)

    return inputs.type(torch.float32)

def load_cmvn(cmvn_file):
    with open(cmvn_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                add_shift_line = line_item[3 : (len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                rescale_line = line_item[3 : (len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue
    means = np.array(means_list).astype(np.float32)
    vars = np.array(vars_list).astype(np.float32)
    cmvn = np.array([means, vars])
    cmvn = torch.as_tensor(cmvn, dtype=torch.float32)
    return cmvn

def compute_fbank_sense_voice(sample,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=1.0,
                  window_type='hamming',
                  lfr_m=1,
                  lfr_n=1,
                  cmvn_file=None):
    """ Extract fbank

        Args:
            sample: {key, wav, sample_rate, ...}

        Returns:
            {key, feat, wav, sample_rate, ...}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    assert 'key' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    waveform = waveform * (1 << 15)
    #print(f"2 waveform: {waveform.shape}")
    # Only keep key, feat, label
    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      energy_floor=0.0,
                      window_type=window_type,
                      sample_frequency=sample_rate,
                      snip_edges=True)
    #print(f"2 mat: {mat.shape}")
    mat = apply_lfr(mat, lfr_m, lfr_n)
    #print(f"22 mat: {mat.shape}")
    if cmvn_file != None:
        cmvn = load_cmvn(cmvn_file)
        mat = apply_cmvn(mat, cmvn)
    #print(f"23 mat: {mat}")
    sample['feat'] = mat
    return sample

def add_frame(sample,
              language="auto",
              use_itn=False,
              textnorm=None,
              embed=None):
    #此代码用于在输入前加入四个标签帧
    from torch import nn
    lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
    textnorm_dict = {"withitn": 14, "woitn": 15}
    input_size = len(sample['feat'][0])
    #print(f"2 fbank——speech: {sample['feat']}")
    # embed = torch.nn.Embedding(
    #     7 + len(lid_dict) + len(textnorm_dict), input_size
    # )
    # for param in embed.parameters():
    #     param.requires_grad = False

    language_query = embed(
        torch.LongTensor([[lid_dict[language] if language in lid_dict else 0]])
    )#生成的张量为（1,1，560）

    language_query = language_query.reshape(1,input_size)
    
    if textnorm is None:
        textnorm = "withitn" if use_itn else "woitn"
    textnorm_query = embed(
        torch.LongTensor([[textnorm_dict[textnorm]]])
    ).reshape(1,input_size)

    speech = torch.cat((textnorm_query, sample['feat']), dim=0)

    event_emo_query = embed(torch.LongTensor([[1, 2]])).reshape(2,input_size)
    #print(f"2 event_emo_query: {event_emo_query}")
    input_query = torch.cat((language_query, event_emo_query), dim=0)

    speech = torch.cat((input_query, speech), dim=0)
    sample['feat'] = speech
    return sample
