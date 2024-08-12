## map & dir
map_path=$1
#map_path="/mnt/cfs/SPEECH/hupeng/oworkdir/audio_book_process/maps/mul/500_600s.txt"
output='/mnt/cfs/SPEECH/data/tts/Audio_Book/part_00/seperate_data'
#out_put='/mnt/cfs/SPEECH/hupeng/oworkdir/audio_book_process/test_data/sep'

## parameters
#gpu='0'
gpu_count=$(nvidia-smi --list-gpus | wc -l)
if [ $? -ne 0 ]; then
    echo "nvidia-smi command failed."
    exit 1
fi
gpu=""
for (( i=0; i<$gpu_count; i++)); do
	if [ -z "$gpu" ]; then
		gpu="$i"
	else
		gpu="$gpu,$i"
	fi
done
echo "Using gpus $gpu"
batch_size=2
save_audio=true
port=29052

echo "Seperation Begin ......"
CUDA_VISIBLE_DEVICES=$gpu accelerate launch --num_processes $gpu_count --num_machines 1 --main_process_port $port ./tal_audio/run_seperate.py \
	--map_path $map_path \
	--save_audio $save_audio \
	--batch_size $batch_size \
	--output $output

echo "Seperation End ......"