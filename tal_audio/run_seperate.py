import warnings
warnings.filterwarnings('ignore')

import os
import sys
sys.path.append('tal_audio')
import argparse
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from dataset.base_dataset import BaseDataset, BaseCollator
from models.seperate.seperate import Seperate

def main(args):
    accelerator = Accelerator(device_placement=True) # device_placement='ddp'
    seperator = Seperate(args)
    
    dict_path = '/mnt/cfs/SPEECH/hupeng/git_loc_workspace/tal_audio/tal_audio/pretrained_models/wenet/cn/lang_char.txt'
    dataset = BaseDataset(dataset_path=args.map_path, 
                          dict_path=dict_path, 
                          label=False, 
                          merge_channel=False,
                          target_sampling_rate=seperator.samplerate, 
                          target_channel=seperator.audio_channels)
    
    collate = BaseCollator(cfg=args, audio_padding=False)
    data_loader = DataLoader(dataset, 
                             batch_size=args.batch_size, 
                             num_workers=8,
                             collate_fn=collate,
                             pin_memory=True)
    
    data_loader, seperator = accelerator.prepare(data_loader, seperator)
    
    batch_count = 0
    start_time = time.time()
    
    for data in data_loader:
        for batch_idx in tqdm(range(len(data["utt"]))):
            utt = data["utt"][batch_idx]
            fpath = data["audio_path"][batch_idx]
            audio = data["audio"][batch_idx]
            # save vocals
            dir_list = os.path.dirname(fpath).split('/')[-2:]
            dir_name = '/'.join(dir_list)
            output_subdir = os.path.join(args.output, dir_name)
            os.makedirs(output_subdir, exist_ok=True)
            save_path = os.path.join(output_subdir, utt+".mp3")
            if os.path.exists(save_path):
                continue
            try:
                tracks = seperator.separate_audio(audio, shifts=1, num_workers=0, progress=False)
                audio_data = tracks["vocals"].detach().cpu()
                seperator.save_audio(save_path, audio_data[0:1,:])
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
        
        batch_count += 1
        if batch_count % 100 == 0:
            print(f"batch_count = {batch_count}")
    
    print(f'总耗时：{time.time()-start_time}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map_path", type=str, default="/mnt/cfs/SPEECH/hupeng/audio_ppl/data_list/test_xk.list", help="输入的map地址")
    parser.add_argument("-s", "--save_audio", type=str, default="false", help="是否保存音频[true,false]")

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument("-o", "--output", type=str, default="./output", help="保存结果数据位置")

    args = parser.parse_args()
    
    main(args)