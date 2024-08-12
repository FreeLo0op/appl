import os
import re
import sys
import numpy as np
import warnings
from tqdm.rich import tqdm
warnings.filterwarnings("ignore")

from tal_audio.utils.audio_process import Audio_PPL

if __name__ == '__main__':
    appl = Audio_PPL()
    
    data_list = sys.argv[1]
    root_dir = sys.argv[2]
    fo_file = sys.argv[3]
    
    # fo_file = r'/mnt/cfs/SPEECH/hupeng/oworkdir/audio_book_process/part_00/part_00_segments_2.list'
    # root_dir = r'/mnt/cfs/SPEECH/data/tts/Audio_Book/part_00/seperate_data'
    # data_list = r'/mnt/cfs/SPEECH/hupeng/oworkdir/audio_book_process/part_00/part_00.list'
    
    fo = open(fo_file, 'w', encoding='utf8')
    fo.write(f'utt\taudio\tstart_time\tend_time\tduration\n')
    
    lines = open(data_list, 'r', encoding='utf8').readlines()
    
    for i in tqdm(  
                    range(len(lines)), 
                    desc="Segments Merge and Boundary Extension",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                ):
        try:
            line = lines[i].strip().split('\t')
            utt = line[0]
            audio_path = os.path.join(root_dir, line[2])
            duration = np.float16(line[-1])
            
            embedding_path = re.sub(r'\.mp3$', r'.pt', audio_path)
            audio_infos = appl.load_embeddings(embedding_path)
            ori_times, merged_segs = appl.segment_merge(audio_infos)
            segments = appl.boundary_extension(merged_segs, duration)
            for j in range(len(segments)):
                utt_id = f'{utt}_{j+1}'
                start_time, end_time = segments[j][0], segments[j][1]
                duration = end_time - start_time
                fo.write(f'{utt_id}\t{audio_path}\t{start_time}\t{end_time}\t{duration}\n')
        except Exception as e:
            continue
    fo.close()
        
        


    