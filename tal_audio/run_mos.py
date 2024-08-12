import warnings
warnings.filterwarnings('ignore')

import os
import glob
import sys
sys.path.append('tal_audio')
import argparse
import time
import torch
import concurrent.futures
from tqdm import tqdm

from models.mos.mos import ComputeScore

SAMPLING_RATE = 16000

def main(args):
    # Init MOS object
    dnsmos_dir = "./tal_audio/pretrained_models/mos/DNSMOS/"
    primary_model_path = os.path.join(dnsmos_dir, 'sig_bak_ovr.onnx')
    compute_score = ComputeScore(primary_model_path)

    # load map
    audio_clips_list = []
    with open(args.map_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            line_splits = line.strip().split('\t')
            audio_clips_list += [line_splits[1]]
    print(f"Load {args.map_path}: Done!")

    start_time = time.time()
    desired_fs = SAMPLING_RATE
    mos_res = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {executor.submit(compute_score, clip, desired_fs): clip for clip in audio_clips_list}
        for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            clip = future_to_url[future]
            #try:
            data = future.result()
            #except Exception as exc:
            #    print('%r generated an exception: %s' % (clip, exc))
            #else:
            mos_res.append(data)

    # Write to map
    with open(args.output, 'w', encoding='utf-8') as fp:
        for res in mos_res:
            audio_path = res['filename']
            utt = os.path.basename(audio_path).split('.')[0]
            sig = f"{res['SIG']:.3f}"
            bak = f"{res['BAK']:.3f}"
            fp.write(utt+'\t'+audio_path+'\t'+sig+'\t'+bak+'\n')
    
    print(f'总耗时：{time.time()-start_time}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ###参数
    parser.add_argument("-m", "--map_path", type=str, default="./segments.map", help="输入的map地址")
    
    ###共有参数参数
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument("-o", "--output", type=str, default="./out.map", help="保存结果数据路径")

    args = parser.parse_args()
    
    main(args)