## map & dir
map_path="/mnt/cfs/SPEECH/data/tts/Audio_Book/data/segments/peiqi/info.map"
output='/mnt/cfs/SPEECH/data/tts/Audio_Book/data/ns/peiqi/'

## parameters
gpu='0,1'
batch_size=1
save_audio=true
realtime=false
port=29052

echo "NS Begin ......"

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port $port ./tal_audio/run_ns.py \
        --map_path $map_path \
        --save_audio $save_audio \
        --batch_size $batch_size \
        --output $output \
        --realtime $realtime

python3 -u ./scripts/get_snr.py \
        -i "$output" \
        -m $map_path \
        -p "*.pt" \
        -o "$output"/info.map

echo "NS End ......"
