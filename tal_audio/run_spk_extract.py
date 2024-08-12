import warnings
warnings.filterwarnings('ignore')

import os
import sys
sys.path.append('tal_audio')
import argparse
import time
import torch
import torchaudio
from torch.utils.data import DataLoader
from accelerate import Accelerator

from dataset.base_dataset import BaseDataset, BaseCollator
import wespeaker


class Spk_Extract:
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()

        self.extractor = wespeaker.load_model('chinese')
            
    def forward(self, data):
        all_result_dict = {}

        # only support batch_size = 1
        audio_path = data["audio_path"][0]
        opath = data["opath"][0]
        embedding = self.extractor.extract_embedding(audio_path)

        # get output dir
        spk_subdir = os.path.dirname(opath)
        if not os.path.exists(spk_subdir):
            os.makedirs(spk_subdir)
        
        torch.save(embedding, opath)

        return all_result_dict

def main(args):
    align = Spk_Extract(args)
    accelerator = Accelerator() # device_placement='ddp'
    
    dict_path = './tal_audio/pretrained_models/wenet/cn/lang_char.txt'
    dataset = BaseDataset(dataset_path=args.map_path, dict_path=dict_path, label=False)
    collate = BaseCollator(cfg=args, audio_padding=False)
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
        all_result_dict = align.forward(data)
        
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"batch_count = {batch_count}")
    
    print(f'总耗时：{time.time()-start_time}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ###参数
    parser.add_argument("-m", "--map_path", type=str, default="./segments.map", help="输入的map地址")
    parser.add_argument("-s", "--save_audio", type=str, default="false", help="是否保存音频[true,false]")

    ###共有参数参数
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument("-o", "--output", type=str, default="./output", help="保存结果数据位置")
    parser.add_argument('--gpu', type=int, default=0, help='gpu id for this rank, -1 for cpu')

    args = parser.parse_args()
    
    main(args)