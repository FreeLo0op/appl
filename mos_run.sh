#!/bin/bash

## map & dir
map_path="/mnt/cfs/SPEECH/zhangxinke1/work/audio/resources/tal_mos/tal.part4"
num_parts=1
output='/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/mos/part4/'

echo "$map_path"

## parameters
gpu='0'
batch_size=1
port=29055 # reserved

# 检查文件是否存在
if [ ! -f "$map_path" ]; then
    echo "Error: File '$map_path' not found!"
    exit 1
fi

# 计算每个部分应该包含的行数
total_lines=$(wc -l < "$map_path")
lines_per_part=$((total_lines / num_parts))
[ $((total_lines % num_parts)) -ne 0 ] && ((lines_per_part++)) # 如果不能整除，每个部分的行数加1

# 打乱文件的行并保存到临时文件
shuffled_file="shuffled_"
shuf "$map_path" > "$shuffled_file"

# 分割文件
split -l "$lines_per_part" "$shuffled_file" part_

# # 清理临时打乱的文件
# rm "$shuffled_file"

echo "MOS Begin ......"

for part_file in part_*; do  
    outpath="$output"/"$part_file"
    echo "$part_file"
    echo "$outpath"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u ./tal_audio/run_mos.py \
        --map_path "$part_file" \
        --batch_size $batch_size \
        --output $outpath &
done

# 等待所有后台进程完成
wait

echo "MOS End ......"
