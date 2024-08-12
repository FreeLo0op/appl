import time
import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm
import numpy as np
from multiprocessing import Process, set_start_method
from wespeaker.cli.speaker import Speaker
from wespeaker.diar.extract_emb import subsegment
import torchaudio

class tal_Speaker(Speaker):
    
    def __init__(self, model_dir: str):
        super().__init__(model_dir)
    
    def set_gpu(self, device_ids):
        available_gpus = torch.cuda.device_count()
        print(f'Available GPUs: {available_gpus}')
        if isinstance(device_ids, int):
            device_ids = [device_ids]
        if any(d >= available_gpus for d in device_ids):
            raise ValueError(f"Invalid device id(s) specified: {device_ids}. Available GPUs: {available_gpus}.")
        if device_ids:
            self.device = torch.device(f'cuda:{device_ids[0]}')
        else:
            self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        if len(device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

    def extract_embedding_feats(self, fbanks, batch_size, subseg_cmn):
        fbanks_array = np.stack(fbanks)
        if subseg_cmn:
            fbanks_array = fbanks_array - np.mean(
                fbanks_array, axis=1, keepdims=True)
        fbanks_array = torch.from_numpy(fbanks_array).to(self.device)
        
        embeddings = None
        for i in range(0, fbanks_array.shape[0], batch_size):
            batch_feats = fbanks_array[i:i + batch_size]
            batch_embs = self.model(batch_feats)
            batch_embs = batch_embs[-1] if isinstance(batch_embs,
                                                      tuple) else batch_embs
            batch_embs = batch_embs.detach().cpu().half()
            
            if embeddings is None:
                embeddings = torch.empty((0, batch_embs.shape[1]), dtype=torch.float16)
            
            embeddings = torch.cat((embeddings, batch_embs), dim=0)
        return embeddings

    def diarize(self, audio_path: str, utt: str = "unk"):
        pcm, sample_rate = torchaudio.load(audio_path, normalize=False)
        # 1. vad
        vad_segments = self.vad.get_speech_timestamps(
            audio_path,
            return_seconds=True
        )

        # 2. extact fbanks
        subsegs, subseg_fbanks = [], []
        window_fs = int(self.diar_window_secs * 1000) // self.diar_frame_shift
        period_fs = int(self.diar_period_secs * 1000) // self.diar_frame_shift
        for item in vad_segments:
            begin, end = item['start'], item['end']
            if end - begin >= self.diar_min_duration:
                begin_idx = int(begin * sample_rate)
                end_idx = int(end * sample_rate)
                tmp_wavform = pcm[0, begin_idx:end_idx].unsqueeze(0).to(
                    torch.float)
                fbank = self.compute_fbank(
                    tmp_wavform,
                    sample_rate=sample_rate,
                    cmn=False
                    )
                tmp_subsegs, tmp_subseg_fbanks = subsegment(
                    fbank=fbank,
                    seg_id="{:08d}-{:08d}".format(int(begin * 1000),
                                                  int(end * 1000)),
                    window_fs=window_fs,
                    period_fs=period_fs,
                    frame_shift=self.diar_frame_shift)
                subsegs.extend(tmp_subsegs)
                subseg_fbanks.extend(tmp_subseg_fbanks)

        # 3. extract embedding
        embeddings = self.extract_embedding_feats(
            subseg_fbanks,
            self.diar_batch_size,
            self.diar_subseg_cmn
        )
        
        # 4. compute vad time
        vad_res = []
        for subseg in subsegs:
            begin_ms, end_ms, begin_frames, end_frames = subseg.split('-')
            begin = (int(begin_ms) +
                     int(begin_frames) * self.diar_frame_shift) / 1000.0
            end = (int(begin_ms) +
                   int(end_frames) * self.diar_frame_shift) / 1000.0
            vad_res.append([begin, end])
        return vad_res, embeddings

def process_batch(batch, model, output_dir):
    print('------Processing Batch------')
    for i in tqdm(range(len(batch)), desc='Processing Batch'):
        item = batch[i]
    #for item in tqdm(batch, desc="Processing Batch"):
        try:
            #key, audio_file, save_pt = item.strip().split('\t')
            item = item.strip().split('\t')
            audio_file = os.path.join(output_dir, item[2])
            save_pt = audio_file.replace('.mp3', '.pt')
            if os.path.exists(save_pt):
                continue
            key = item[0]
            
            vad_res, embeddings = model.diarize(audio_file)
            res = {}
            for i in range(len(vad_res)):
                res[f'begin_end_{i}'] = torch.tensor(vad_res[i], dtype=torch.float16)
                res[f'embedding_{i}'] = embeddings[i]
            
            torch.save(res, save_pt)
        except Exception as e:
            print(f'Error processing {audio_file}: {str(e)}')

def process_batches(batches, model, output_dir):
    print('------Processing Batches------')
    for batch in batches:
        process_batch(batch, model, output_dir)

if __name__ == '__main__':
    data_list = sys.argv[1]
    output_dir = sys.argv[2]
    set_start_method('spawn')
    
    model_dir = r'/mnt/cfs/SPEECH/hupeng/speaker_verification/wespeaker/models/eres2net_cn_commom_200k'
    
    # Create two model instances
    model1 = tal_Speaker(model_dir)
    model2 = tal_Speaker(model_dir)
    
    # Set GPUs for both models
    model1.set_gpu([0])
    model2.set_gpu([1])

    batch_size = 4  # Adjust the batch size as needed
    #data_list = r'/mnt/cfs/SPEECH/hupeng/oworkdir/audio_book_process/selected_2wh_books/set1_single.list'
    # output_dir = r'/mnt/cfs/SPEECH/data/tts/Audio_Book/part_00/seperate_data'
    os.makedirs(output_dir, exist_ok=True)
    time_start = time.time()
    
    with open(data_list, 'r', encoding='utf8') as fin:
        lines = fin.readlines()
        mid_index = len(lines) // 2
        
        # Split the data into two halves for the two models
        lines1 = lines[:mid_index]
        lines2 = lines[mid_index:]
        
        batches1 = [lines1[i:i+batch_size] for i in range(0, len(lines1), batch_size)]
        batches2 = [lines2[i:i+batch_size] for i in range(0, len(lines2), batch_size)]
        print('-----Finished Spliting Data------')
        
        # Create two processes
        p1 = Process(target=process_batches, args=(batches1, model1, output_dir))
        p2 = Process(target=process_batches, args=(batches2, model2, output_dir))
        
        # Start the processes
        p1.start()
        p2.start()
        
        # Wait for the processes to finish
        p1.join()
        p2.join()

    time_end = time.time()
    total_time = round((time_end - time_start) / 3600, 2)
    print(f'Spend {total_time} Hours to Process {len(lines)} Audio Data')
