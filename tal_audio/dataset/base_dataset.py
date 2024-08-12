import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class BaseDataset(torch.utils.data.Dataset):    
    def __init__(self, dataset_path:str, dict_path:str, label=False, merge_channel=True, target_sampling_rate=16000, target_channel=1):
        self.merge_channel = merge_channel
        self.target_sampling_rate = target_sampling_rate
        self.target_channel = target_channel

        self.utt_list, self.audio_list, self.text_list, self.dur_list, self.opath_list = [], [], [], [], []
        with open(dataset_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                line_splits = line.strip().split('\t')
                if len(line_splits) < 2:
                    print(f"Error: {line}")
                    continue
                
                self.utt_list += [line_splits[0]]
                self.audio_list += [line_splits[1]]
                
                if label == 'true':
                    self.text_list += [line_splits[2]]
                else:
                    self.text_list += ['<unk>']
                
                self.dur_list += [float(line_splits[3])]
                self.opath_list += [line_splits[4]]
        
        print(f"Loading dataset [{dataset_path}] is Done!")

        # load dict
        self.char_dict = {}
        with open(dict_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                line_splits = line.strip().split()
                assert len(line_splits) == 2
                self.char_dict[line_splits[0]] = int(line_splits[1])
        
        print(f"Loading dict [{dict_path}] is Done!")

    def get_token(self, text):
        label = [self.char_dict[x] for x in text.strip().split()] # process <unk>
        label = torch.Tensor(label)    
        return label
    
    def load_audio(self, audio_path):
        sampling_rate = self.target_sampling_rate

        audio, fs = torchaudio.load(audio_path)

        if audio.size(0) > 1 and self.merge_channel:
            audio = audio.mean(dim=0, keepdim=True)
        
        if audio.size(0) >=self.target_channel:
            audio = audio[:self.target_channel,:]
        else:
            audio_list = [audio[0:1,:] for _ in range(self.target_channel)]
            audio = torch.concat(audio_list, dim=0)

        if fs != sampling_rate:
            transform = torchaudio.transforms.Resample(orig_freq=fs,
                                                       new_freq=sampling_rate)
            audio = transform(audio)
            fs = sampling_rate

        return audio.squeeze(0), fs

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        single_feature = dict()

        utt = self.utt_list[idx]
        single_feature["utt"] = utt
        
        text = self.text_list[idx]
        single_feature["text"] = text

        label = self.get_token(text)
        label_len = len(label)
        single_feature["label"] = label
        single_feature["label_len"] = label_len

        audio_path = self.audio_list[idx]
        single_feature["audio_path"] = audio_path

        try:
            audio, fs = self.load_audio(audio_path)
        except:
            print("Load Audio Error: "+utt+'\t'+audio_path)
            audio = torch.zeros((16000,), dtype=torch.float)
            fs = -1
            
        audio_len = audio.shape[0]
        single_feature["audio"] = audio
        single_feature["audio_len"] = audio_len
        single_feature["fs"] = fs
        
        _, audio_format = os.path.splitext(audio_path)
        single_feature["audio_format"] = audio_format

        single_feature["dur"] = self.dur_list[idx]
        single_feature["opath"] = self.opath_list[idx]

        return single_feature

class BaseCollator(object):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg, audio_padding=True):
        self.cfg = cfg
        self.audio_padding = audio_padding

    # def __call__(self, batch):
    def __call__(self, raw_batch):
        packed_batch_features = dict()

        # audio: [b, t]
        # frame_pitch, frame_energy: [1, T]
        # target_len: [1]
        # spk_id: [b, 1]
        # mask: [b, t, 1]

        batch = []
        for feature in raw_batch:
            if feature["fs"] > -1: 
                batch += [feature]

        for key in batch[0].keys():
            if key in ["utt", "text", "fs", "audio_path", "audio_format", "opath", "dur"]:
                packed_batch_features[key] = [b[key] for b in batch]
            elif key == "label_len":
                packed_batch_features["label_len"] = torch.LongTensor(
                    [b["label_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["label_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["label_mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "audio_len":
                packed_batch_features["audio_len"] = torch.LongTensor(
                    [b["audio_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["audio_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["audio_mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            else:
                values = [b[key] for b in batch]
                if self.audio_padding:
                    packed_batch_features[key] = pad_sequence(
                        values, batch_first=True, padding_value=0
                    )
                else:
                    packed_batch_features[key] = values
                
        return packed_batch_features


if __name__=="__main__":
    dataset_path = "./resources/align_example.map"
    dict_path = './tal_audio/pretrained_models/wenet/cn/lang_char.txt'
    dataset = BaseDataset(dataset_path=dataset_path, dict_path=dict_path, label='true')
    collate = BaseCollator(cfg=None)
    data_loader = DataLoader(dataset, 
                             batch_size=2, 
                             num_workers=2, 
                             collate_fn=collate)

    for data in data_loader:
        for wav in data['audio']:
            print(wav.shape)
    
    print('Done!')
 
