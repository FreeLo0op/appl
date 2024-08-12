## map & dir
map_path=$1

## parameters

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
batch_size=64
port=29052

echo "ASR Begin ......"
CUDA_VISIBLE_DEVICES=$gpu accelerate launch --num_processes $gpu_count --num_machines 1 --main_process_port $port ./tal_audio/run_asr.py \
	--map_path $1 \
	--batch_size $batch_size \

echo "ASR End ......"