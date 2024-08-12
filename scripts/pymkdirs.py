import os
import argparse


# mkdir subdirs before multi process
def main_v1(map_path, output): # old version: to be deleated later
    subdir_list = []
    with open(map_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            line_splits = line.strip().split('\t')
            dir_list = os.path.dirname(line_splits[1]).split('/')[-2:]
            # dir_list = os.path.dirname(line_splits[1]).split('/')[-3:]
            dir_name = '/'.join(dir_list)
            output_subdir = os.path.join(output, dir_name)
            subdir_list += [output_subdir]

    subdir_uniq = list(set(subdir_list))
    print("subdir_list = ", len(subdir_uniq))
    print(subdir_uniq[0])

    count = 0
    for output_subdir in subdir_uniq:
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        count += 1
        if count % 5000 == 0:
            print("count = ", count)

    print('Done')

def main(map_path):
    subdir_list = []
    with open(map_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            line_splits = line.strip().split('\t')
            output_subdir = os.path.dirname(line_splits[4]) # 4 means the filename of output file
            subdir_list += [output_subdir]

    subdir_uniq = list(set(subdir_list))
    print("subdir_list = ", len(subdir_uniq))
    print(subdir_uniq[0])

    count = 0
    for output_subdir in subdir_uniq:
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        count += 1
        if count % 5000 == 0:
            print("count = ", count)

    print('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map_path", type=str, default="./segments.map", help="输入的map地址")
    # parser.add_argument("-o", "--output", type=str, default="./output", help="保存结果数据位置")

    args = parser.parse_args()
    
    if os.path.exists(args.map_path):
        # main(args.map_path, args.output)
        main(args.map_path)
    else:
        print(f"Error: {args.map_path} not exist!")
