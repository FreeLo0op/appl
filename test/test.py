# %%
import os
import time
import shutil
from textgrid import TextGrid, IntervalTier
import torch
import torchaudio
# torch.set_num_threads(1)


class VAD:
    def __init__(self, output) -> None:
        self.fs = 16000
        USE_ONNX = False
        model_load_type = "local" # option: local or github

        # load model
        if model_load_type == "local":
            # local dir
            self.model, self.utils = torch.hub.load(repo_or_dir='tal_audio/pretrained_models/vad/silero-vad',
                                        model='silero_vad',
                                        source='local',
                                        force_reload=False,
                                        onnx=USE_ONNX)
        else:
            self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                        model='silero_vad',
                                        force_reload=False,
                                        onnx=USE_ONNX)

        self.model.to('cuda')
        self.model.eval()

        # self.segments_dir = os.path.join(args.output, 'segments')
        self.segments_dir = os.path.join(output, 'segments')
        if not os.path.exists(self.segments_dir):
            os.mkdir(self.segments_dir)

    def timestamp_normalize(self, tss):
        beg_end_list = []
        for i in range(0, len(tss), 2):
            beg_end_list += [(tss[i]['start'], tss[i+1]['end'])]

        # paras
        max_speech_duration_s = 15.
        min_speech_duration_s = 2.
        sil_gap = 0.5

        cur_start = beg_end_list[0][0]
        cur_end = beg_end_list[0][1]

        merge_list = []
        for item in beg_end_list[1:]:
            dur = item[1] - item[0]
            if item[0] - cur_end < sil_gap:
                if cur_end - cur_start + dur < max_speech_duration_s:
                    cur_end = item[1]
                else:
                    merge_list += [(cur_start, cur_end)]
                    cur_start = item[0]
                    cur_end = item[1]
            else:
                if cur_end - cur_start < min_speech_duration_s and min_speech_duration_s <= cur_end - cur_start + dur <= max_speech_duration_s:
                    cur_end = item[1]
                else:
                    merge_list += [(cur_start, cur_end)]
                    cur_start = item[0]
                    cur_end = item[1]

        # add head offset
        offset = 0.1
        last_end = 0.
        norm_list = []
        for (start, end) in merge_list:
            if start - offset >= last_end:
                norm_list += [(start-offset, end)]
            else:
                norm_list += [(start, end)]
            
            # update last_end
            last_end = end
        
        return norm_list
    
    def generate_textgrid(self, norm_list, tg_path, maxTime):
        endTime = maxTime
        tg = TextGrid(maxTime=maxTime)

        tier_graphemes = IntervalTier(name="vad")
        for (start, end) in norm_list:
            minTime = start # secs
            maxTime = end # secs
            grapheme_interval = [minTime, maxTime, '1']
            tier_graphemes.add(*grapheme_interval)

        if maxTime < endTime:
            grapheme_interval = [maxTime, endTime, ""]
            tier_graphemes.add(*grapheme_interval)

        tg.append(tier_graphemes)
        tg.write(tg_path)
    
    def split_audio(self, audio, norm_list, utt, audio_path, save_dir):
        audio = audio.cpu()
        output_list = []
        for idx, (start, end) in enumerate(norm_list):
            split_utt = utt+'_'+str(idx)
            fpath = os.path.join(save_dir, split_utt+'.flac')
            split_start = (int)(start * self.fs)
            split_end = (int)(end * self.fs)
            splitted = (audio[split_start:split_end]*32767).short()
            torchaudio.save(fpath, splitted.unsqueeze(0), self.fs, bits_per_sample=16)
            output_list += [split_utt+'\t'+fpath+'\t'+audio_path+'\t'+str(start)+'\t'+str(end)+'\t'+str(end-start)+'\n']
        
        save_map_path = os.path.join(save_dir, utt+'.map')
        with open(save_map_path, 'w', encoding='utf-8') as fp:
            for line in output_list:
                fp.write(line)
        
        return save_map_path

    def forward(self, audio_path):
        # get function
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = self.utils
        
        # load audio
        wav = read_audio(audio_path, sampling_rate=self.fs)
        maxTime = len(wav)/self.fs
        wav = wav.to('cuda')

        # normalize to +/- 1.0
        wav = wav/wav.abs().max()

        vad_iterator = VADIterator(self.model)

        tss = []
        window_size_samples = 1536 # number of samples in a single audio chunk
        total_frames = len(wav)//window_size_samples

        start_time = time.time()

        frame_idx = 0
        for i in range(0, len(wav), window_size_samples):
            chunk = wav[i: i+window_size_samples]
            if len(chunk) < window_size_samples:
                break
            
            with torch.no_grad():
                speech_dict = vad_iterator(chunk, return_seconds=True)
            
            if speech_dict:
                tss += [speech_dict]
            
            frame_idx += 1
            if frame_idx % 2000 == 0:
                print(f"frame is {frame_idx}/{total_frames} = {100*frame_idx/total_frames}%")

        vad_iterator.reset_states() # reset model states after each audio

        print(f'Total time of VAD is {time.time()-start_time}')

        # merge vad intervals and normalize
        if 'end' not in tss[-1].keys():
            tss += [{'end':maxTime}]
        
        print(audio_path, tss)

        # merge vad intervals and normalize
        norm_list = self.timestamp_normalize(tss)

        # format output dir
        utt = os.path.basename(audio_path).split('.')[0]
        dir_name = os.path.dirname(audio_path).split('/')[-1]

        segment_subdir = os.path.join(self.segments_dir, dir_name)
        audio_subdir = os.path.join(segment_subdir, utt)
        if not os.path.exists(audio_subdir):
            os.makedirs(audio_subdir)
        
        tg_path = os.path.join(audio_subdir, utt+'.TextGrid')
        self.generate_textgrid(norm_list, tg_path, maxTime)

        # split audio
        # save_map_path = self.split_audio(wav, norm_list, utt, audio_path, audio_subdir)
        shutil.copy(audio_path, os.path.join(audio_subdir, os.path.basename(audio_path)))
        print("split_audio: done")

        processed = dir_name+'/'+utt
        # return processed, save_map_path
        return processed, None

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("-a", "--audio_path", type=str, default="./test.flac", help="input audio file path")
#     parser.add_argument("-o", "--output", type=str, default="./align_output", help="output directory")

#     args = parser.parse_args()
    
#     vad_obj = VAD(args)
#     # wav_path = "/mnt/cfs/SPEECH/zhangxinke1/work/audio/resources/audios/17_926361.wav"
#     wav_path = "/mnt/cfs/SPEECH/data/tts/tal_lessons/audio/8/295518.mp3"
    
#     processed, save_map_path = vad_obj.forward(wav_path)
#     print(f"Audio = {processed}, Map = {save_map_path}, Done!")


#%%
import argparse

wav_scp_path = "/mnt/cfs/SPEECH/zhangwenkai/FunASR/workspace/check/wav.scp"
out_dir = "/mnt/cfs/SPEECH/zhangxinke1/work/audio/vad_output"


# parser = argparse.ArgumentParser()
# parser.add_argument("-o", "--output", type=str, default=out_dir, help="output directory")
# args = parser.parse_args()
# print(args.output)

SAMPLING_RATE = 16000

model, utils = torch.hub.load(repo_or_dir='tal_audio/pretrained_models/vad/silero-vad',
                                        model='silero_vad',
                                        source='local',
                                        force_reload=False,
                                        onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

output_list = []
with open(wav_scp_path, 'r', encoding='utf-8') as fp:
    for line in fp:
        wav_path = line.strip()

        wav = read_audio(wav_path, sampling_rate=SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE,
                                                  return_seconds=True)

        print(wav_path, speech_timestamps)
        output_list += [wav_path+'\t'+str(speech_timestamps[-1]['end'])+'\n']

# %%
# with open("/mnt/cfs/SPEECH/zhangxinke1/work/wav.scp", 'w', encoding='utf-8') as fp:
#     for line in output_list:
#         fp.write(line)
# print('Done!')


# %%
# debug mos
import argparse
import os

import librosa
import numpy as np
import numpy.polynomial.polynomial as poly
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from requests import session
from tqdm import tqdm

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

class ComputeScore:
    def __init__(self, primary_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path)

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, fpath, sampling_rate, is_personalized_MOS=False):
        aud, input_fs = sf.read(fpath)
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, input_fs, fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        max_length = 2*len_samples
        if len(audio) > max_length:
            audio = audio[:max_length]
        
        print(f"audio is {audio.shape}, {len(audio)}")

        num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            oi = {'input_1': input_features}
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,is_personalized_MOS)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)

        clip_dict = {'filename': fpath, 'len_in_sec': actual_audio_len/fs, 'sr':fs}
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        return clip_dict
    
    
#%%
audio_path = "/mnt/cfs/SPEECH/zhangxinke1/work/audio/old/828040_11.flac"
# audio_path = "/mnt/cfs/SPEECH/zhangxinke1/work/audio/old/828069_4.flac"
# audio_path = "/mnt/cfs/SPEECH/zhangxinke1/work/audio/old/666385_3.flac"

dnsmos_dir = "/mnt/cfs/SPEECH/zhangxinke1/work/audio/tal_audio/pretrained_models/mos/DNSMOS/"
primary_model_path = os.path.join(dnsmos_dir, 'sig_bak_ovr.onnx')

compute_score = ComputeScore(primary_model_path)

score = compute_score(audio_path, SAMPLING_RATE)

print(f"SIG = {score['SIG']:.2f}, BAK={score['BAK']:.2f}")


# %%
from funasr import AutoModel

model = AutoModel(model="damo/speech_rwkv_bat_asr-zh-cn-16k-aishell1-vocab4234-pytorch", 
                  model_revision="v2.0.2",
                 )

#%%
res = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")
print(res)



# %%
import os

# extract error map
map_dir = "/mnt/cfs/SPEECH/zhangxinke1/work/audio/resources/tal_asr/"
subsets = ['large.map', 'small.map']

err_dict = {}
with open(os.path.join(map_dir, 'asr_todo.txt'), 'r', encoding='utf-8') as fp:
    for line in fp:
        err_dict[line.strip()] = 1

output_list = []
for subset in subsets:
    with open(os.path.join(map_dir, subset), 'r', encoding='utf-8') as fp:
        for line in fp:
            fpath = line.strip().split('\t')[1]
            if fpath in err_dict:
                output_list += [line]
print('Done')
print(len(output_list))

# %%
# with open(os.path.join(map_dir, 'asr_todo.map'), 'w', encoding='utf-8') as fp:
#     for item in output_list:
#         fp.write(item)

# %%
import torch
import torchaudio

wav_path = "./resources/audios/xiaosi/06d7bad92bc1474480548619e5e1b7c2.wav"

audio, fs = torchaudio.load(wav_path)

# %%
p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1]
print(p)

#%%
print(audio.shape)
audio = torch.nn.functional.pad(audio, (0, p), mode='constant').data
print(audio.shape)

y_pad = torch.nn.functional.pad(audio, (40, 40), "reflect")
print(y_pad.shape)

# %%
audio, fs = torchaudio.load(wav_path)

audio_ext = torch.concat([audio, audio], dim=0)

print(audio_ext.shape)
audio_ext = torch.nn.functional.pad(audio_ext, (0, p), mode='constant')
print(audio_ext.shape)

audio_ext = torch.nn.functional.pad(audio_ext, (40, 40), "reflect")
print(audio_ext.shape)

# %%
x = [5.7, 4.2, 3.6, 2.8, 3.7, 5.4, 2.0, 1.5, 3.9, 1.8, 4.2, 1.3, 5.7, 8.2, 6.1, 6.7, 1.7, 4.1, 7.5, 6.0]


# %%
import torch
import torchaudio

file_list = ["/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_311.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_246.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_18.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_298.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_562.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_293.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_303.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_61.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_174.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_256.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_116.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_313.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_371.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_587.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_190.flac",
"/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/9/240089/240089_84.flac"]


# %%
for item in file_list:
    
    x,fs = torchaudio.load(item)
    print(item, x.shape)


# %%
import os

xpad_audio_map_path = "/mnt/cfs/SPEECH/data/asr/xpad/online_data_1kw/wav.scp"

output_list = []
with open(xpad_audio_map_path, 'r', encoding='utf-8') as fp:
    for line in fp:
        line_splits = line.strip().split()
        utt = line_splits[0]
        audio_path = line_splits[1]

        # print(utt)
        # print(audio_path)
        dur = 8.0
        output_list += [utt+'\t'+audio_path+'\t'+'<unk>'+'\t'+str(dur)+'\n']
        # break

# %%
with open('/mnt/cfs/SPEECH/zhangxinke1/work/audio/resources/xpad/xpad_1300w.map', 'w', encoding='utf-8') as fp:
    for line in output_list:
        fp.write(line)

print('Done')

# %%
# test paraformer
from funasr import AutoModel

model = AutoModel(model="paraformer-zh", model_revision="v2.0.4")

#%%
hotword = '小思 小思小思 葛红 abc reading'
res = model.generate(input="/mnt/cfs/SPEECH/data/asr/xpad/audio/2024-01-10/split/1/c61631c2a7814ff1aee4884d0046f179.wav",
                     hotword=hotword,)
print(res)


# %%
# wenetspeech
import os

wenetspeech_map_path = "/mnt/cfs/SPEECH/hupeng/wenetspeech/result/wenet_map_info"

output_list = []
with open(wenetspeech_map_path, 'r', encoding='utf-8') as fp:
    for line in fp:
        line_splits = line.strip().split()
        utt = line_splits[0]
        audio_path = line_splits[1]
        text = line_splits[2]
        
        dur = 8.0
        output_list += [utt+'\t'+audio_path+'\t'+text+'\t'+str(dur)+'\n']
        # break


# %%
# o_map_path = '/mnt/cfs/SPEECH/zhangxinke1/work/audio/resources/wenetspeech/wenetspeech.map'
# with open(o_map_path, 'w', encoding='utf-8') as fp:
#     for line in output_list:
#         fp.write(line)

# print('Done')


# %%
import os

text_map_path = "/mnt/cfs/SPEECH/data/asr/xpad/xpad_funasr.map"

output_list = []
with open(text_map_path, 'r', encoding='utf-8') as fp:
    for line in fp:
        line_splits = line.strip().split('\t')
        if len(line_splits) == 2:
            utt = os.path.basename(line_splits[0]).split('.')[0]
            text = line_splits[1]
            output_list += [utt+' '+text+'\n']
        # else:
        #     print(line)

print(f"xpad utt number is {len(output_list)}")
print('Done')


# %%
import os

output_list = []
count = 0
with open('/mnt/cfs/SPEECH/zhangxinke1/work/audio/resources/wenetspeech/wenetspeech.map', 'r', encoding='utf-8') as fp:
    for line in fp:
        line_splits = line.strip().split('\t')

        utt = line_splits[0]
        audio_path = line_splits[1]

        text_path = audio_path.replace('/mnt/cfs/SPEECH/hupeng/wenetspeech/result/data/wenetspeech_ext/audio/', '/mnt/cfs/SPEECH/data/asr/WenetSpeech/text/')
        text_path = text_path.replace('.flac', '.txt')

        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.readline().strip()
            
            output_list += [utt+' '+text+'\n']

            count += 1
            if count % 10000 == 0:
                print(f"count = {count}")
        
        # break
            
print('Done')


# %%
# o_map_path = '/mnt/cfs/SPEECH/data/asr/WenetSpeech/wenetspeech_0407.txt'
# with open(o_map_path, 'w', encoding='utf-8') as fp:
#     for line in output_list:
#         fp.write(line)

# print('Done')
# %%
import os
import torch
import torchaudio

#%%
peiqi_dir = "/mnt/cfs/SPEECH/data/tts/Audio_Book/ertonggushi/小猪佩奇1-9季_中文版_天马座动画/"
mp3_list = [x for x in os.listdir(peiqi_dir) if x.endswith('.mp3')]


output_list = []
for mp3_name in mp3_list:
    mp3_path = os.path.join(peiqi_dir, mp3_name)
    audio, fs = torchaudio.load(mp3_path)
    print(audio.shape, fs)

    output_list += [mp3_name.split('.mp3')[0]+'\t'+mp3_path+'\t'+'<unk>'+'\t'+str(audio.shape[-1]/fs)+'\n']
    # break

# %%
with open('/mnt/cfs/SPEECH/zhangxinke1/work/audio/resources/peiqi/peiqi_mp3.map', 'w', encoding='utf-8') as fp:
    for item in output_list:
        fp.write(item)

print('Done')

# %%
dur = 0.0
for item in output_list:
    dur += float(item.split('\t')[-1])

print(dur/3600.0)
# %%
import os
import random
import hashlib
import torch
import torchaudio


def generate_unique_shortname(orig_name):
    hash_object = hashlib.sha1(orig_name.encode())
    hex_dig = hash_object.hexdigest()
    unique_filename = hex_dig[:16].lower()
    return unique_filename


def try_get_short_name(orig_name, occupy_dict, max_tries=10):
    count = 0
    while(count < max_tries):
        if count == 0:
            rand_str = ''
        else:
            rand_str = str(random.randint(1, 32767))
        
        new_name = generate_unique_shortname(orig_name+rand_str)

        if new_name not in occupy_dict:
            return new_name

        count += 1
    return new_name+'_'+rand_str


def preprocess(map_path, output_dir, num_wavs_per_dir=1000, verbose=True):
    file_count = 0
    utt_dict = {}
    with open(map_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            line_splits = line.strip().split('\t')
            utt = line_splits[0]
            audio_path = line_splits[1]

            subdir_new = os.path.join(output_dir, 'part_'+str(file_count//num_wavs_per_dir))
            if not os.path.exists(subdir_new):
                os.makedirs(subdir_new)
            
            audio_name_new = try_get_short_name(utt, utt_dict)
            if audio_name_new not in utt_dict:
                audio_path_new = os.path.join(subdir_new, audio_name_new+'.mp3')

                audio, fs = torchaudio.load(audio_path)
                if audio.size(0) > 1:
                    audio = audio[0:1,:]
                
                dur = audio.shape[-1]/fs
                dir_name = os.path.dirname(audio_path).split('/')[-1]
                utt_dict[audio_name_new] = audio_name_new+'\t'+audio_path_new+'\t'+'<unk>'+'\t'+str(dur)+'\t'+dir_name+'/'+utt+'\n'
                
                torchaudio.save(audio_path_new, audio, fs, bits_per_sample=32)
            else:
                print("Error: ", utt+'\t'+audio_path)
            
            file_count += 1

            if verbose and file_count % 10 == 0:
                with open(os.path.join(output_dir, "info.map"), 'w', encoding='utf-8') as fp:
                    for item in utt_dict:
                        fp.write(utt_dict[item])

                print(f"file_count = {file_count}")
                
    
    return utt_dict        


# %%
input_map_path = "/mnt/cfs/SPEECH/zhangxinke1/work/audio/resources/peiqi/peiqi_mp3.map"
output_dir = "/mnt/cfs/SPEECH/data/tts/Audio_Book/data/raw/peiqi/"

preprocess(input_map_path, output_dir, num_wavs_per_dir=100)


# %%
import json
label_path = "/mnt/cfs/SPEECH/hupeng/data_xk/data_16k_2.json"

with open(label_path, 'r', encoding='utf-8') as f:
    data_dict = json.load(f)


# %%
phones = {}
for utt in data_dict:
    for phn in data_dict[utt]['phn'].strip().split():
        phones[phn] = 1

phones_list = [x for x in phones]
phones_list.sort()

for idx, item in enumerate(phones_list):
    print(str(idx)+'\t'+item)

print('\nTotal Phones = ', len(phones_list))

with open('./align_output/phones.dict', 'w', encoding='utf-8') as fp:
    fp.write(str(0)+'\t'+'sil'+'\n')
    fp.write(str(1)+'\t'+'<eos>'+'\n')

    for idx, item in enumerate(phones_list):
        if item != 'sil':
            fp.write(str(idx+2)+'\t'+item+'\n')
    
print('Done')


# %%
import os
import glob

segment_dir = "/mnt/cfs/SPEECH/data/tts/Audio_Book/data/segments/"

# 搜索当前目录及其所有子目录中的所有.map文件
map_list = [file for file in glob.glob(os.path.join(segment_dir, '**/*.map'), recursive=True)]

print(map_list[0])
print(len(map_list))

# %%
import shutil
import torch
import torchaudio

data_map = "/mnt/cfs/SPEECH/data/tts/Audio_Book/data/ns/peiqi/info.map.good"
output_dir = "/mnt/cfs/SPEECH/data/tts/Audio_Book/data/sel_v1/"

with open(data_map, 'r', encoding='utf-8') as fp:
    for line in fp:
        line_splits = line.strip().split('\t')
        utt_start = float(line_splits[3])
        utt_end = float(line_splits[4])

        x,fs = torchaudio.load(line_splits[2])
        wav_dur = x.shape[-1]/fs

        if utt_start > 17. and utt_end < (wav_dur - 14.5):
            audio_name = os.path.basename(line_splits[1])
            dst_path = os.path.join(output_dir, audio_name)
            shutil.copy(line_splits[1], dst_path)

print('Done')

# %%
