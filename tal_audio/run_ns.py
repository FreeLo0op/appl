import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio
import os
import sys
sys.path.append('tal_audio')
import argparse
import time
from torch.utils.data import DataLoader
from accelerate import Accelerator

from dataset.base_dataset import BaseDataset, BaseCollator
from models.ns.ns import CkptInferenceSnri


class NS_Impl:
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()

        self.ns_obj = CkptInferenceSnri(self.args)

        self.output = args.output
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)
    
    def save_audio(self, batch_audio, batch_utt, batch_len, batch_path, batch_snr):
        batch_audio = batch_audio.detach().cpu().unsqueeze(dim=1)

        for idx, utt in enumerate(batch_utt):
            # get output dir
            dir_list = os.path.dirname(batch_path[idx]).split('/')[-2:]
            dir_name = '/'.join(dir_list)
            output_subdir = os.path.join(self.output, dir_name)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            save_audio_path = os.path.join(output_subdir, utt+".flac")
            
            audio = (batch_audio[idx,...]*32767).short() # float to short
            
            if batch_len[idx] < audio.shape[-1]:
                torchaudio.save(save_audio_path, audio[:,:batch_len[idx]], sample_rate=16000)
            else:
                torchaudio.save(save_audio_path, audio, sample_rate=16000)
            
            snr = batch_snr[idx].cpu()
            torch.save(snr, os.path.join(output_subdir, utt+".pt"))

    @torch.no_grad()        
    def forward(self, data):
        all_result_dict = {}

        audio_denoised, audio_snr = self.ns_obj.main(data["audio"], data["audio_len"])

        self.save_audio(audio_denoised, data["utt"], data["audio_len"], data["audio_path"], audio_snr)

        return all_result_dict

def main(args):
    run_obj = NS_Impl(args)
    accelerator = Accelerator()
    
    dict_path = './tal_audio/pretrained_models/wenet/cn/lang_char.txt'
    dataset = BaseDataset(dataset_path=args.map_path, dict_path=dict_path, label=False)
    collate = BaseCollator(cfg=args, audio_padding=True)
    data_loader = DataLoader(dataset, 
                             batch_size=args.batch_size, 
                             num_workers=8,
                             collate_fn=collate,
                             pin_memory=True)
    
    data_loader, run_obj = accelerator.prepare(data_loader, run_obj)
    
    batch_count = 0
    start_time = time.time()
    for data in data_loader:
        try:
            all_result_dict = run_obj.forward(data)
        except:
            for item in data["audio_path"]:
                print("Error: "+item)
        
        batch_count += 1
        if batch_count % 100 == 0:
            print(f"batch_count = {batch_count}")
    
    print(f'总耗时：{time.time()-start_time}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map_path", type=str, default="./segments.map", help="输入的map地址")
    parser.add_argument("-s", "--save_audio", type=str, default="false", help="是否保存音频[true,false]")

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument("-o", "--output", type=str, default="./output", help="保存结果数据位置")
    parser.add_argument("--realtime", type=str, default="false", help="是否启用流式")

    args = parser.parse_args()
    
    main(args)