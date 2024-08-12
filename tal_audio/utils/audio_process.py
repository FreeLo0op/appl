import os
import torch
import numpy as np
from collections import defaultdict
from pydub import AudioSegment
from tqdm import tqdm

class Audio_PPL():
    def __init__(self) -> None:
        self.time_threshold = 0.55
        self.cossim_threshold = 0.4
        self.extension_threshold = 0.5
        
        self.f_max_segment_time = 15
        self.max_segment_time = 20
            
    def load_embeddings(self, pt_file_path):
        try:
            data = torch.load(pt_file_path)
            infos = defaultdict(list)
            for key, value in data.items():
                infos[key].append(value)
            return infos
        except Exception as e:
            print(f'Error loading {pt_file_path}: {str(e)}')
            return None
    
    def normalize_vector(self, vector):
        max_val = np.max(np.abs(vector))
        if max_val == 0:
            return vector
        return vector / max_val
    
    def cosine_similarity(self, embedding1:np.array, embedding2:np.array) -> float:
        embedding1 = np.array(embedding1, dtype=np.float32)
        embedding2 = np.array(embedding2, dtype=np.float32)
        
        # 对嵌入向量进行归一化处理
        #embedding1 = normalize_vector(embedding1)
        #embedding2 = normalize_vector(embedding2)
        
        norm_embedding1 = np.linalg.norm(embedding1)
        norm_embedding2 = np.linalg.norm(embedding2)
        
        if norm_embedding1 == 0 or norm_embedding2 == 0:
            return 0.0
        
        dot_product = np.dot(embedding1, embedding2)
        cosine_sim = dot_product / (norm_embedding1 * norm_embedding2)
        
        return cosine_sim
    
    def cs_threshold(self, cosine_sim):
        if cosine_sim < self.cossim_threshold:
            return False
        else:
            return True
    
    def sil_threshold(self, t1:float, t2:float) -> bool:
        sil_interval = float(t2-t1)
        if sil_interval > self.time_threshold:
            return True
        else:
            return False
    
    def segment_merge(self, audio_infos: defaultdict[list]) -> list[list]:
        infos_len = len(audio_infos) // 2
        times, merge = [], []
        if infos_len == 1:
            audio_infos['begin_end_0'][0].numpy()
        else:
            for i in range(0, infos_len-1):
                begin_end_h = audio_infos[f'begin_end_{i}']
                begin_h, end_h = begin_end_h[0].numpy()
                times.append([begin_h, end_h])
                
                embedding_h = audio_infos[f'embedding_{i}'][0]
                if isinstance(embedding_h, torch.Tensor):
                    embedding_h = embedding_h.numpy()
                
                begin_end_b = audio_infos[f'begin_end_{i+1}']
                begin_b, end_b = begin_end_b[0].numpy()
                
                embedding_b = audio_infos[f'embedding_{i+1}'][0]
                if isinstance(embedding_b, torch.Tensor):
                    embedding_b = embedding_b.numpy()
                sil_interval = begin_b - end_h
                
                if sil_interval > self.time_threshold:
                    # sil 持续时间大于 slef.sil_threshold 不合并
                    merge.append(False)
                elif sil_interval < 0:
                    # 两段音频有交集，直接合并
                    merge.append(True)
                else:
                    # sil 小于阈值的情况
                    if self.sil_threshold(begin_h, end_h):
                        cs = self.cosine_similarity(embedding1=embedding_h, embedding2=embedding_b)
                        if self.cs_threshold(cs):
                            # 余弦相似度大于阈值，合并
                            merge.append(True)
                        else:
                            # 余弦相似度小于阈值，不合并
                            merge.append(False)
                    else:
                        # 段落持续时间小于阈值，直接合并，不计算余弦相似度
                        merge.append(True)
            times.append([begin_b, end_b])
        merged_segs = [times[0]]
        #print("Times ===> ",times)
        max_interval = 0
        for i in range(len(merge)):
            seg = times[i+1]
            if merge[i]:
                tmp_max = [merged_segs[-1][0], max(merged_segs[-1][1], seg[1])]
                interval = tmp_max[1] - tmp_max[0]
                
                if self.max_segment_time > interval:
                    merged_segs[-1] = tmp_max
                else:
                    merged_segs.append(seg)
                #merged_segs[-1] = [merged_segs[-1][0], max(merged_segs[-1][1], seg[1])]
            else:
                merged_segs.append(seg)
            #max_interval = max(max_interval, (merged_segs[-1][1] - merged_segs[-1][0]))
        #print("Merged Segs ===> ", merged_segs)
        #print("Max Interval ===> ", max_interval)
        return times, merged_segs
    
    def boundary_extension(self, segments:list[list], duration:float) -> list[list]:
        seg_len = len(segments)
        if seg_len == 1:
            return segments
        for i in range(seg_len-1):
            mid_point = (segments[i+1][0] - segments[i][1]) / 2
            if mid_point > self.extension_threshold:
                extend_time = self.extension_threshold
            else:
                extend_time = mid_point
            segments[i][1] += extend_time
            segments[i+1][0] -= extend_time
        
        start_time = segments[0][0]
        if start_time > self.extension_threshold:
            segments[0][0] -= self.extension_threshold
        else:
            segments[0][0] /= 2

        dur = duration - segments[-1][1]
        if dur > self.extension_threshold:
            segments[-1][-1] += self.extension_threshold
        else:
            segments[-1][-1] += dur / 2

        return segments
    
    def audio_split(self, audio:str, segments_list:list[list], save_dir):
        if not os.path.exists(audio):
            raise ValueError(f'{audio} not exists')
        
        tmp = audio.strip().split('/')[-1]
        son_dir, audio_format = tmp.split('.')
        
        save_infos = os.path.join(save_dir, son_dir, 'segment.list')        
        save_dir = os.path.join(save_dir, son_dir, 'audios')
        os.makedirs(save_dir, exist_ok=True)
        fo = open(save_infos, 'w', encoding='utf8')
        
        
        audio = AudioSegment.from_file(audio, format=audio_format)

        
        # sample_rate = audio.frame_rate
        # channels = audio.channels
        for i in range(len(segments_list)):
            segment = segments_list[i]
            start_time, end_time = segment
            fo.write(f'{son_dir}_{i+1}.{audio_format}\t{start_time}\t{end_time}\n')
            
            segment = audio[float(start_time)*1000: float(end_time)*1000]
            savepath = os.path.join(save_dir, f'{son_dir}_{i+1}.{audio_format}')
            segment.export(savepath, format=audio_format)
        fo.close()

if __name__ == '__main__':
    
    audio_list = r'/mnt/cfs/SPEECH/hupeng/audio_ppl/data_list/audio_book_2w_shuf_100.list'
    save_dir = r'/mnt/cfs/SPEECH/hupeng/audio_ppl/test_audios_3'
    root_dir = r'/mnt/cfs/SPEECH/data/tts/Audio_Book/part_00/seperate_data'
    
    audio_dur = {}
    with open(r'/mnt/cfs/SPEECH/hupeng/oworkdir/audio_book_process/audio_80w_info.list', 'r', encoding='utf8') as fin:
        lines = fin.readlines()[1:]
        for line in lines:
            line = line.strip().split(' ', maxsplit=4)
            if len(line) == 5:
                audio_dur[line[0]] = float(line[-1])
                    
    appl = Audio_PPL(audio_dur)
    
    with open(audio_list, 'r', encoding='utf8') as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines)), desc='Audio Book PPL'):
            line = lines[i]
            line = line.strip().split('\t')
            key = line[0]
            audio = os.path.join(root_dir, line[2])
            pt = audio.replace('.mp3', '.pt')
            print(f'====== Processing {key} ======')
            audio_infos = appl.load_embeddings(pt)
            ori_times, merged_segs = appl.segment_merge(audio_infos)
            print('Original Segments Length ===>', len(ori_times))
            #print('Original Segments ===>',ori_times)
            #print('Merged Segments ===>',merged_segs)
            segments = appl.boundary_extension(merged_segs, float(audio_dur[key]))
            #print('Expand Boundary Segments ===>',segments)
            print('Expand Boundary Segments Length ===>', len(segments))
            print()
            
            #appl.audio_split(audio, segments, save_dir)

    