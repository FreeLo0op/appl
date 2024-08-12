#!/bin/bash

# 获取GPU数量
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "CUDA_VISIBLE_DEVICES num is ${NUM_GPUS}"
# 输入数据列表文件
INPUT_DATA_LIST="/mnt/cfs/SPEECH/dengtengyue1/wenet/examples/xpad/s0/test/segments_01.json"

# 临时目录用于存储中间结果
TEMP_DIR="./temp_results2"
mkdir -p $TEMP_DIR

# 最终结果文件
FINAL_RESULT_FILE="final_result2.txt"
> $FINAL_RESULT_FILE # 清空或创建最终结果文件

# 读取输入数据列表并计算每个子列表的大小
TOTAL_LINES=$(wc -l < $INPUT_DATA_LIST)
LINES_PER_GPU=$(( (TOTAL_LINES + NUM_GPUS - 1) / NUM_GPUS ))

split_input_files=()

# 分割输入数据列表
for (( i=0; i<$NUM_GPUS; i++ ))
do
    split_file="${TEMP_DIR}/input_part_${i}.txt"
    start_line=$(( i * LINES_PER_GPU + 1 ))
    end_line=$(( (i + 1) * LINES_PER_GPU ))
    sed -n "${start_line},${end_line}p" $INPUT_DATA_LIST > $split_file
    split_input_files+=($split_file)
done
echo "数据列表分割完成，接下来进行推理"
# 并行在多个GPU上进行推理
# 存储进程ID的数组
pids=()
for (( i=0; i<$NUM_GPUS; i++ ))
do
    CUDA_VISIBLE_DEVICES=$i python sensevoice_test.py --input ${split_input_files[$i]} --output ${TEMP_DIR}/result_${i}.txt &
    pid=$!
    echo "Started process on GPU $i with PID $pid"
    pids+=($pid)
done

# 等待所有子进程完成，并同时检查其退出状态
all_success=true
for pid in "${pids[@]}"; do
    if wait $pid; then
        echo "Process $pid completed successfully."
    else
        echo "Process $pid failed."
        all_success=false
    fi
done

# 合并所有GPU的推理结果到最终结果文件
for (( i=0; i<$NUM_GPUS; i++ ))
do
    cat ${TEMP_DIR}/result_${i}.txt >> $FINAL_RESULT_FILE
done

echo "All results have been merged into $FINAL_RESULT_FILE"