import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from funasr import AutoModel
from accelerate import Accelerator
import time

from utils.dataloader import AudioDataset, collate_fn
import warnings
warnings.filterwarnings('ignore')

def save_results(res, audio_files):
    for item, audio_file in zip(res, audio_files):
        text = item['text']
        if text:
            key = item['key']
            save_dir = audio_file.replace('seperate_data', 'asr_res').replace('.mp3', '')
            save_pt = os.path.join(save_dir, f'{key}.pt')
            if not os.path.exists(save_pt):
                torch.save(item, save_pt)

def main(args):
    
    accelerator = Accelerator(device_placement=True)
    acdevice = accelerator.device
    model = AutoModel(
        model='/mnt/cfs/SPEECH/common/funasr_model/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    )
    model.model.to(acdevice)
    model = accelerator.prepare(model)
    
    df = pd.read_csv(args.map_path, sep='\t')
    audios = set(df['audio'])
    for audio in audios:
        save_dir = audio.replace('seperate_data', 'asr_res').replace('.mp3', '')
        os.makedirs(save_dir, exist_ok=True)
    num_splits = args.num_splits
    split_dfs = np.array_split(df, num_splits)
    
    total_items = sum(len(split_df) for split_df in split_dfs)
    pbar = tqdm(total=total_items, desc='ASR Processing', leave=False)
    
    for i in range(num_splits):
        audio_dataset = AudioDataset(split_dfs[i], audio_load=True, sort_order=None)
        audio_loader = DataLoader(audio_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=64)
        data_loader = accelerator.prepare(audio_loader)
        
        for batch in data_loader:
            try:
                utts, audio_segments, audio_files, _ = batch
                
                res = model.inference(
                    input=audio_segments,
                    batch_size=len(utts),
                    key=list(utts)
                )
                save_results(res, audio_files)
                pbar.update(len(audio_files)*2)
                
            except Exception as e:
                print(e)
                continue
    pbar.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--map_path", type=str, default="/mnt/cfs/SPEECH/hupeng/audio_ppl/data_list/test_dataloader_1w.list", help="输入的map地址")
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--num_splits', type=int, default=5000, help='划分成多个子任务')

    args = parser.parse_args()
    main(args)

