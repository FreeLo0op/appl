#%%
import os
import torch
import torchaudio
import shutil

format = "scp" # scp or dir

#%%
# format: dir
if format == "dir":
    print("format: dir")

    data_dir = "./resources/"
    audio_dir = os.path.join(data_dir, 'audios')

    new_list = []
    for item in os.listdir(audio_dir):
        utt = item.split('.')[0]
        text = item.split('_')[0]
        fpath = os.path.join(audio_dir, item)
        audio, fs = torchaudio.load(fpath)
        duration = audio.shape[-1]/fs

        new_list += [utt+'\t'+fpath+'\t'+text+'\t'+f"{duration:.3f}"+'\n']

    with open(os.path.join(data_dir, 'align_example.map'), 'w', encoding='utf-8') as fp:
        for item in new_list:
            fp.write(item)

elif format == 'scp':
    print("format: scp") # fname fpath

    # format: scp
    scp_path = "/mnt/cfs/SPEECH/zhangwenkai/data_clean/wav.scp.500"

    data_dir = "./resources/"
    audio_dir = os.path.join(data_dir, 'xiaosi_audios')

    if not os.path.exists(audio_dir):
        os.mkdir(audio_dir)

    new_list = []
    with open(scp_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            line_splits = line.strip().split()
            fname = line_splits[0]
            fpath = line_splits[1]

            utt = fname.split('.')[0]

            fpath_new = os.path.join(audio_dir, fname)
            shutil.copy(fpath, fpath_new)

            audio, fs = torchaudio.load(fpath)
            duration = audio.shape[-1]/fs

            text = " "

            new_list += [utt+'\t'+fpath_new+'\t'+text+'\t'+f"{duration:.3f}"+'\n']


    with open(os.path.join(data_dir, 'align_xiaosi.map'), 'w', encoding='utf-8') as fp:
        for item in new_list:
            fp.write(item)

print("Done!")

# %%
