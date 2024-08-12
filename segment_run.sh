## map & dir
map_path="/mnt/cfs/SPEECH/data/tts/Audio_Book/data/sep/peiqi/info.map"
output='/mnt/cfs/SPEECH/data/tts/Audio_Book/data/segments/peiqi/'

## parameters
gpu='0,1'
batch_size=1
save_audio=true

echo "Segment Begin ......"
CUDA_VISIBLE_DEVICES=$gpu accelerate launch ./tal_audio/run_segment.py --map_path $map_path \
        --save_audio $save_audio \
        --batch_size $batch_size \
        --output $output

python3 -u ./scripts/get_files.py \
        -i "$output" \
        -p "*.map" \
        -o "$output"/info.map

echo "Segment End ......"
