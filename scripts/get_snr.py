import os
import glob
import argparse
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="./input", help="input directory")
    parser.add_argument("-p", "--pattern", type=str, default="*.pt", help="pattern, eg, .pt|.npy")
    parser.add_argument("-m", "--map_path", type=str, default="./input.map", help="输入的map地址")
    parser.add_argument("-o", "--output", type=str, default="./info.map", help="输出的map地址")

    args = parser.parse_args()
    
    search_pattern = '**/'+args.pattern
    snr_list = [file for file in glob.glob(os.path.join(args.input, search_pattern), recursive=True)]

    snr_dict = {}
    for snr_path in snr_list:
        utt = os.path.basename(snr_path).split('.')[0]
        snr_dict[utt] = torch.load(snr_path).numpy()
    
    output_list = []
    good_list = []
    with open(args.map_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            utt = line.strip().split('\t')[0]
            if utt in snr_dict:
                o_line = line.strip()+'\t'+str(snr_dict[utt][0])
                output_list += [o_line]

                if snr_dict[utt] > 25.0:
                    good_list += [o_line]
    
    with open(args.output, 'w', encoding='utf-8') as fp:
        for line in output_list:
            fp.write(line+'\n')
    
    with open(args.output+'.good', 'w', encoding='utf-8') as fp:
        for line in good_list:
            fp.write(line+'\n')
    
    print('Done!')