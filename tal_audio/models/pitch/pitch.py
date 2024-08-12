import os
import torch
import torchaudio
import numpy as np
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT


def get_yaapt_f0(audio, sr=16000, interp=False):
    to_pad = int(20.0 / 1000 * sr) // 2
    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0) 
        pitch = pYAAPT.yaapt(basic.SignalObj(y_pad, sr), 
                             **{'frame_length': 20.0, 'frame_space': 5.0, 'nccf_thresh1': 0.25, 'tda_frame_length': 25.0, 'f0_max': 1000.0})
        f0s.append(pitch.samp_interp[None, None, :] if interp else pitch.samp_values[None, None, :])

    return np.vstack(f0s)


class Pitch:
    def __init__(self, args) -> None:
        self.args = args
        self.method = "yaapt"

        self.pad_1 = 1280
        self.downsample_ratio = self.pad_1//16

    def load_audio(self, audio_path, sampling_rate=16000):
        audio, fs = torchaudio.load(audio_path)

        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)

        if fs != sampling_rate:
            transform = torchaudio.transforms.Resample(orig_freq=fs,
                                                       new_freq=sampling_rate)
            audio = transform(audio)
            fs = sampling_rate

        return audio, fs
    
    def save_pitch(self, pitch, opath):
        # get output dir
        # dir_list = os.path.dirname(audio_path).split('/')[-2:]
        # dir_name = '/'.join(dir_list)
        # pitch_subdir = os.path.join(self.pitch_dir, dir_name)
        # utt = os.path.basename(audio_path).split('.')[0]
        pitch_subdir = os.path.dirname(opath)
        if not os.path.exists(pitch_subdir):
            os.makedirs(pitch_subdir)
        
        # save_path = os.path.join(pitch_subdir, utt+".pt")
        save_path = opath
        torch.save(pitch, save_path)
    
    def __call__(self, paths, interp=False):
        audio_path, opath = paths
        audio, sr = self.load_audio(audio_path, sampling_rate=16000)

        p = (audio.shape[-1] // self.pad_1 + 1) * self.pad_1 - audio.shape[-1]
        audio = torch.nn.functional.pad(audio, (0, p), mode='constant')
        audio_numpy = audio.numpy()

        status = False
        try:
            f0 = get_yaapt_f0(audio_numpy, sr=sr, interp=interp)
            status = True
        except:
            f0 = np.zeros((1, 1, audio_numpy.shape[-1] // self.downsample_ratio))

        f0 = torch.FloatTensor(f0.astype(np.float32).squeeze(0))

        # save
        self.save_pitch(f0, opath)

        return status