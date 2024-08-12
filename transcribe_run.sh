## map & dir
map_path="/mnt/cfs/SPEECH/data/tts/tal_question/raw/202303/info.map"

# extract_type='mkdirs'
extract_type='trans'

## parameters
gpu='0,1'
batch_size=2
save_audio=false
port=29051

if [ "$extract_type" = "mkdirs" ]; then
    echo "Make directories"
    
    python3 -u ./scripts/pymkdirs.py\
        --map_path $map_path

    echo "Make directories End ......"
fi

if [ "$extract_type" = "trans" ]; then
    echo "Transcribe Begin ......"
    
    CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port $port ./tal_audio/run_transcribe.py \
        --map_path $map_path \
        --save_audio $save_audio \
        --batch_size $batch_size

    echo "Transcribe End ......"
fi
