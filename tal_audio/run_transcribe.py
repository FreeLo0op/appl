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
from models.asr.asr import ASR


class Transcribe:
    def __init__(self,args):
        self.args = args
        self.accelerator = Accelerator()

        self.transcribe = ASR(self.args)

        # load hotword
        if os.path.exists(args.hotword):
            hotword = []
            with open(args.hotword, 'r', encoding='utf-8') as fp:
                for line in fp:
                    hotword += [line.strip().split()[0]]
            self.hotword = ' '.join(hotword)
        else:
            # debug
            self.hotword = ''
        
        print('self.hotword = ', self.hotword)
            
    def forward(self, data):
        all_result_dict = {}
        
        flag = False
        try:
            asr_predict = self.transcribe.forward(data["utt"], 
                                                  data["audio"], 
                                                  data["opath"],
                                                  hotword=self.hotword)
        except:
            flag = True

        if flag:
            for batch_idx in range(len(data["utt"])):
                #try:
                    utt = [data["utt"][batch_idx]]
                    audio = [data["audio"][batch_idx]]
                    opath = [data["opath"][batch_idx]]
                    audio_path = [data["audio_path"][batch_idx]]

                    asr_predict = self.transcribe.forward(utt, 
                                                          audio, 
                                                          opath,
                                                          hotword=self.hotword)
                #except:
                #    print("# Error: "+utt[0]+'\t'+audio_path[0])
        print(asr_predict)
        return all_result_dict

def main(args):
    align = Transcribe(args)
    accelerator = Accelerator() # device_placement='ddp'
    
    dict_path = './tal_audio/pretrained_models/wenet/cn/lang_char.txt'
    dataset = BaseDataset(dataset_path=args.map_path, dict_path=dict_path, label=False)
    collate = BaseCollator(cfg=args, audio_padding=False)

    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = DataLoader(dataset, 
                             batch_size=args.batch_size, 
                             num_workers=8,
                             collate_fn=collate,
                             sampler=sampler,
                             shuffle=False,
                             pin_memory=True)
    
    data_loader, align = accelerator.prepare(data_loader, align)
    
    batch_count = 0
    start_time = time.time()
    output_list = []
    for data in data_loader:
        # print("****** data = ", data)
        all_result_dict = align.forward(data)
        
        batch_count += 1
        if batch_count % 500 == 0:
            print(f"batch_count = {batch_count}")
    
    print(f'总耗时：{time.time()-start_time}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ###参数
    parser.add_argument("-m", "--map_path", type=str, default="./segments.map", help="输入的map地址")
    parser.add_argument("-s", "--save_audio", type=str, default="false", help="是否保存音频[true,false]")

    ###共有参数参数
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id for this rank, -1 for cpu')
    parser.add_argument("--hotword", type=str, default="./resources/xpad/hotword.txt", help="热词列表")

    args = parser.parse_args()
    
    main(args)