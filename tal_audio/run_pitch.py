import warnings
warnings.filterwarnings('ignore')

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import sys
sys.path.append('tal_audio')
import argparse
import time
import torch
import concurrent.futures
from tqdm import tqdm

from models.pitch.pitch import Pitch

def main(args):
    # Init Pitch object
    compute_pitch = Pitch(args)

    # pitch_dir = args.output
    # if not os.path.exists(pitch_dir):
    #     os.makedirs(pitch_dir, exist_ok=True)

    # load map
    audio_clips_list = []
    with open(args.map_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            line_splits = line.strip().split('\t')
            # audio_clips_list += [line_splits[1]]
            audio_clips_list += [(line_splits[1], line_splits[4])]
    print(f"Load {args.map_path}: Done!")

    start_time = time.time()
    pitch_res = []
    # option 1: single process
    for clip in audio_clips_list:
        status = compute_pitch.__call__(clip)
        pitch_res.append(status)
    
    print(f'总耗时：{time.time()-start_time}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map_path", type=str, default="./segments.map", help="输入的map地址")
    parser.add_argument("-w", "--max_workers", type=int, default=16, help="并行进程数量")

    args = parser.parse_args()
    
    main(args)