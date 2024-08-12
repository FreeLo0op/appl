import os
import glob
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="./output", help="input directory")
    parser.add_argument("-p", "--pattern", type=str, default="*.map", help="pattern, eg, .map|.wav")
    parser.add_argument("-o", "--output", type=str, default="./info.map", help="输出的map地址")

    args = parser.parse_args()
    
    search_pattern = '**/'+args.pattern
    map_list = [file for file in glob.glob(os.path.join(args.input, search_pattern), recursive=True)]

    output_list = []
    for map_path in map_list:
        with open(map_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                output_list += [line.strip()]
    
    with open(args.output, 'w', encoding='utf-8') as fp:
        for line in output_list:
            fp.write(line+'\n')
    
    print('Done!')