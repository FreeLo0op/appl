# options: mkdirs, spk, w2v, pitch
extract_type=$1

## map & dir
map_path=$2

echo $extract_type
echo $map_path

## parameters
gpu='0,1'
batch_size=8
save_audio=false
port=29053

echo "Extract Begin ......"

if [ "$extract_type" = "mkdirs" ]; then
    echo "Make directories"
    python3 -u ./scripts/pymkdirs.py\
        --map_path $map_path
fi

if [ "$extract_type" = "spk" ]; then
    echo "Speaker Embedding Extracting"
    CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port $port ./tal_audio/run_spk_extract.py \
        --map_path $map_path \
        --save_audio $save_audio \
        --batch_size $batch_size
fi

if [ "$extract_type" = "w2v" ]; then
    echo "Wav2Vec Embedding Extracting"
    CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port $port ./tal_audio/run_w2v_extract.py \
        --map_path $map_path \
        --save_audio $save_audio \
        --layer 7 \
        --batch_size $batch_size
fi

if [ "$extract_type" = "pitch" ]; then
    # 检查文件是否存在
    if [ ! -f "$map_path" ]; then
        echo "Error: File '$map_path' not found!"
        exit 1
    fi

    tmp_dir="./tmp"
    if [ ! -d "$tmp_dir" ]; then
        mkdir -p "$tmp_dir"
    fi

    # 计算每个部分应该包含的行数
    num_parts=10
    total_lines=$(wc -l < "$map_path")
    lines_per_part=$((total_lines / num_parts))
    [ $((total_lines % num_parts)) -ne 0 ] && ((lines_per_part++)) # 如果不能整除，每个部分的行数加1

    # 打乱文件的行并保存到临时文件
    filename=$(basename "$map_path" .map)
    echo $filename
    shuffled_file="$filename"_shuffled
    shuf "$map_path" > "$tmp_dir"/"$shuffled_file"

    # 分割文件
    split_name="$filename"_part_
    split -l "$lines_per_part" "$tmp_dir"/"$shuffled_file" "$tmp_dir"/"$split_name"

    # for part_file in part_*; do
    for part_file in "$tmp_dir"/"$split_name"*; do
        echo "$part_file"
        
        python3 -u ./tal_audio/run_pitch.py \
            --map_path "$part_file" \
            --max_workers 2 &

    done

    # 等待所有后台进程完成
    wait
fi

echo "Extract End ......"
