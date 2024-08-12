import warnings
warnings.filterwarnings('ignore')

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append('tal_audio')
import argparse
import time
import torch

def save_pitch(data, fpath, output_dir):
    # get output dir
    dir_list = os.path.dirname(fpath).split('/')[-2:]
    dir_name = '/'.join(dir_list)
    pitch_subdir = os.path.join(output_dir, dir_name)
    utt = os.path.basename(fpath).split('.')[0]
    if not os.path.exists(pitch_subdir):
        os.makedirs(pitch_subdir)
    
    save_path = os.path.join(pitch_subdir, utt+".pt")
    torch.save(data, save_path)

def main(args):

    pitch_dir = args.output
    if not os.path.exists(pitch_dir):
        os.makedirs(pitch_dir, exist_ok=True)

    # load map
    audio_clips_list = []
    with open(args.map_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            line_splits = line.strip().split('\t')
            audio_clips_list += [line_splits[1]]
    print(f"Load {args.map_path}: Done!")

    start_time = time.time()
    pitch_res = []
    for clip in audio_clips_list:
        
        pitch_res.append(status)
    
    print(f'总耗时：{time.time()-start_time}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map_path", type=str, default="./segments.map", help="输入的map地址")
    parser.add_argument("-o", "--output", type=str, default="./output", help="保存结果数据位置")
    parser.add_argument("-w", "--max_workers", type=int, default=16, help="并行进程数量")

    args = parser.parse_args()
    
    main(args)