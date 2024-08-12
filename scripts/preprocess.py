import os
import argparse
import random
import hashlib
import torch
import torchaudio


def generate_unique_shortname(orig_name):
    hash_object = hashlib.sha1(orig_name.encode())
    hex_dig = hash_object.hexdigest()
    unique_filename = hex_dig[:16].lower()
    return unique_filename


def try_get_short_name(orig_name, occupy_dict, max_tries=10):
    count = 0
    while(count < max_tries):
        if count == 0:
            rand_str = ''
        else:
            rand_str = str(random.randint(1, 32767))
        
        new_name = generate_unique_shortname(orig_name+rand_str)

        if new_name not in occupy_dict:
            return new_name

        count += 1
    return new_name+'_'+rand_str


def preprocess(map_path, output_dir, num_wavs_per_dir=1000, verbose=True):
    file_count = 0
    utt_dict = {}
    with open(map_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            line_splits = line.strip().split('\t')
            utt = line_splits[0]
            audio_path = line_splits[1]

            subdir_new = os.path.join(output_dir, 'part_'+str(file_count//num_wavs_per_dir))
            if not os.path.exists(subdir_new):
                os.makedirs(subdir_new)
            
            audio_name_new = try_get_short_name(utt, utt_dict)
            if audio_name_new not in utt_dict:
                audio_path_new = os.path.join(subdir_new, audio_name_new+'.mp3')

                audio, fs = torchaudio.load(audio_path)
                
                # Uncomment to save only channel 0
                # if audio.size(0) > 1:
                #     audio = audio[0:1,:]
                
                dur = audio.shape[-1]/fs
                dir_name = os.path.dirname(audio_path).split('/')[-1]
                utt_dict[audio_name_new] = audio_name_new+'\t'+audio_path_new+'\t'+'<unk>'+'\t'+str(dur)+'\t'+dir_name+'/'+utt+'\n'
                
                torchaudio.save(audio_path_new, audio, fs, bits_per_sample=32)
            else:
                print("Error: ", utt+'\t'+audio_path)
            
            file_count += 1

            if verbose and file_count % 10 == 0:
                with open(os.path.join(output_dir, "info.map"), 'w', encoding='utf-8') as fp:
                    for item in utt_dict:
                        fp.write(utt_dict[item])

                print(f"file_count = {file_count}")
    
    # At the end, save the whole dict
    with open(os.path.join(output_dir, "info.map"), 'w', encoding='utf-8') as fp:
        for item in utt_dict:
            fp.write(utt_dict[item])
                
    return utt_dict        


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map_path", type=str, default="./segments.map", help="输入的map地址")
    parser.add_argument("-o", "--output", type=str, default="./output", help="保存结果数据位置")

    args = parser.parse_args()

    utt_dict = preprocess(args.map_path, args.output, num_wavs_per_dir=1000)
    print("Preprocess: Done!")