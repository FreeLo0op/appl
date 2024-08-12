import warnings
warnings.filterwarnings('ignore')

import os
import sys
sys.path.append('tal_audio')
import argparse
import time
import torch
import transformers
from torch.utils.data import DataLoader
from accelerate import Accelerator

from dataset.base_dataset import BaseDataset, BaseCollator


class W2V_Extract:
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mms = transformers.Wav2Vec2ForPreTraining.from_pretrained("/mnt/cfs/SPEECH/zhangxinke1/models/mms-300m")

        # with internet access, can set it to 'facebook/mms-300m'
        # self.mms = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/mms-300m")

        # map model to specific device: cuda/cpu
        self.mms.to(self.device)

        self.pad_1 = 1280
        self.pad_2 = 40

        for param in self.mms.parameters():
            param.requires_grad = False
            param.grad = None
        
        self.mms.eval()
        self.feature_layer = args.layer
    
    def save_w2v(self, batch_w2v, batch_utt, batch_len, batch_path, batch_opath):
        batch_w2v = batch_w2v.detach().cpu().unsqueeze(dim=1)
        
        for idx, utt in enumerate(batch_utt):
            # get output dir
            save_path = batch_opath[idx]
            w2v_subdir = os.path.dirname(save_path)
            if not os.path.exists(w2v_subdir):
                os.makedirs(w2v_subdir)
            
            w2v = batch_w2v[idx,:,:,:batch_len[idx]]
            w2v_half = w2v.half() # in order to save disk, float32 occupies too much
            torch.save(w2v_half, save_path)

    @torch.no_grad()        
    def forward(self, data):
        all_result_dict = {}

        audio = data["audio"]

        p = (audio.shape[-1] // self.pad_1 + 1) * self.pad_1 - audio.shape[-1]
        audio_pad = torch.nn.functional.pad(audio, (0, p), mode='constant')
        audio_pad = torch.nn.functional.pad(audio_pad, (self.pad_2, self.pad_2), "reflect")
        
        audio_p = (data["audio_len"] // self.pad_1 + 1) * self.pad_1 - data["audio_len"]
        audio_len = (data["audio_len"] + audio_p)//(self.pad_1//4)

        mms_outputs = self.mms(audio_pad.squeeze(1), output_hidden_states=True)
        w2v = mms_outputs.hidden_states[self.feature_layer]
        w2v = w2v.permute((0, 2, 1))

        # save w2v
        self.save_w2v(w2v, data["utt"], audio_len, data["audio_path"], data["opath"])

        return all_result_dict

def main(args):
    align = W2V_Extract(args)
    accelerator = Accelerator() # device_placement='ddp'
    
    dict_path = './tal_audio/pretrained_models/wenet/cn/lang_char.txt'
    dataset = BaseDataset(dataset_path=args.map_path, dict_path=dict_path, label=False)
    collate = BaseCollator(cfg=args, audio_padding=True)
    data_loader = DataLoader(dataset, 
                             batch_size=args.batch_size, 
                             num_workers=8,
                             collate_fn=collate,
                             pin_memory=True)
    
    data_loader, align = accelerator.prepare(data_loader, align)
    
    batch_count = 0
    start_time = time.time()
    output_list = []
    for data in data_loader:
        try:
            all_result_dict = align.forward(data)
        except:
            for item in data["audio_path"]:
                print("Error: "+item)
        
        batch_count += 1
        if batch_count % 100 == 0:
            print(f"batch_count = {batch_count}")
    
    print(f'总耗时：{time.time()-start_time}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ###参数
    parser.add_argument("-m", "--map_path", type=str, default="./segments.map", help="输入的map地址")
    parser.add_argument("-s", "--save_audio", type=str, default="false", help="是否保存音频[true,false]")

    ###共有参数参数
    parser.add_argument('--layer', type=int, default=7, help='mms feature layer')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    # parser.add_argument("-o", "--output", type=str, default="./output", help="保存结果数据位置")
    parser.add_argument('--gpu', type=int, default=0, help='gpu id for this rank, -1 for cpu')

    args = parser.parse_args()
    
    main(args)