#%%
# import os
# import torchaudio
# from types import SimpleNamespace
# import subprocess
# import pandas as pd
# from tal_audio.models.asr.asr import ASR

# # %%
# csv_path = "./xiaosi_sample.csv"
# df = pd.read_csv(csv_path, encoding='utf-8')

# # %%
# args = SimpleNamespace(output='./align_output/')
# transcribe = ASR(args)

# # %%
# hotword = "小思小思"
# for idx in range(len(df['audio_url'])):
#     audio_url = df['audio_url'][idx]
#     audio_name = audio_url.strip().split('/')[-1]
#     out_path = './align_output/'
#     audio_path = os.path.join(out_path, audio_name)

#     result = subprocess.run([f'wget -q {audio_url} -P {out_path}'], shell=True)

#     audio_data, fs = torchaudio.load(audio_path)
#     audio_data = audio_data.squeeze(dim=0)

#     asr_predict = transcribe.forward([audio_name.split('.')[0]], [audio_data], [audio_path], hotword=hotword)
#     print(audio_name, '|', df['asr_output_text'][idx], '|', asr_predict[0])

# %%
import os
import torch
import torchaudio

wav_dir = "/mnt/cfs/SPEECH/data/asr/xpad/audio/20240205_20240213/"
wav_scp_path = os.path.join(wav_dir, "wav.scp")
text_path = os.path.join(wav_dir, 'text.online')

wav_dict = {}
with open(wav_scp_path, 'r', encoding='utf-8') as fp:
    for line in fp:
        utt, wav_path = line.strip().split()
        wav_dict[utt] = wav_path

text_dict = {}
with open(text_path, 'r', encoding='utf-8') as fp:
    for line in fp:
        line_splits = line.strip().split()
        utt = line_splits[0]
        text_dict[utt] = ' '.join(line_splits[1:])

print('Load Done')


# %%
output_list = []
count = 0
for utt in wav_dict:
    x, fs = torchaudio.load(wav_dict[utt])
    dur = x.shape[1]/fs
    output_list += [utt+'\t'+wav_dict[utt]+'\t'+text_dict[utt]+'\t'+str(dur)+'\n']
    
    count += 1
    if count % 10000 == 0:
        print(f"count = {count}")


# %%
with open('./20240205_20240213.map', 'w', encoding='utf-8') as fp:
    for line in output_list:
        fp.write(line)

print("Done")