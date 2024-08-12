# %%
import os
import time
from textgrid import TextGrid, IntervalTier
import torch
import torchaudio
# torch.set_num_threads(1) # ToDo: check the function


class VAD:
    def __init__(self, args) -> None:
        self.fs = 16000
        USE_ONNX = False
        model_load_type = "local" # option: local or github

        # load model
        if model_load_type == "local":
            # local dir
            self.model, self.utils = torch.hub.load(repo_or_dir='tal_audio/pretrained_models/vad/silero-vad',
                                        model='silero_vad',
                                        source='local',
                                        force_reload=False,
                                        onnx=USE_ONNX)
        else:
            self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                        model='silero_vad',
                                        force_reload=False,
                                        onnx=USE_ONNX)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        # self.model.to('cuda')
        self.model.eval()
        print("***********: ", self.model)

        self.segments_dir = args.output
        if not os.path.exists(self.segments_dir):
            os.mkdir(self.segments_dir)

    def timestamp_normalize(self, tss):
        beg_end_list = []
        for i in range(0, len(tss), 2):
            beg_end_list += [(tss[i]['start'], tss[i+1]['end'])]

        # paras
        max_speech_duration_s = 15.
        min_speech_duration_s = 2.
        sil_gap = 0.5
        sil_gap_max = 1.0

        cur_start = beg_end_list[0][0]
        cur_end = beg_end_list[0][1]

        merge_list = []
        for item in beg_end_list[1:]:
            dur = item[1] - item[0]
            if item[0] - cur_end < sil_gap:
                if cur_end - cur_start + dur < max_speech_duration_s:
                    cur_end = item[1]
                else:
                    merge_list += [(cur_start, cur_end)]
                    cur_start = item[0]
                    cur_end = item[1]
            else:
                if cur_end - cur_start < min_speech_duration_s and min_speech_duration_s <= cur_end - cur_start + dur <= max_speech_duration_s:
                    if item[0] - cur_end > sil_gap_max:
                        merge_list += [(cur_start, cur_end)]
                        cur_start = item[0]
                        cur_end = item[1]
                    else:
                        cur_end = item[1]
                else:
                    merge_list += [(cur_start, cur_end)]
                    cur_start = item[0]
                    cur_end = item[1]

        # add head offset
        offset = 0.1
        last_end = 0.
        norm_list = []
        for (start, end) in merge_list:
            if start - offset >= last_end:
                norm_list += [(start-offset, end)]
            else:
                norm_list += [(start, end)]
            
            # update last_end
            last_end = end
        
        return norm_list
    
    def generate_textgrid(self, norm_list, tg_path, maxTime):
        endTime = maxTime
        tg = TextGrid(maxTime=maxTime)

        tier_graphemes = IntervalTier(name="vad")
        for (start, end) in norm_list:
            minTime = start # secs
            maxTime = end # secs
            grapheme_interval = [minTime, maxTime, '1']
            tier_graphemes.add(*grapheme_interval)

        if maxTime < endTime:
            grapheme_interval = [maxTime, endTime, ""]
            tier_graphemes.add(*grapheme_interval)

        tg.append(tier_graphemes)
        tg.write(tg_path)
    
    def split_audio(self, audio, norm_list, utt, audio_path, save_dir):
        audio = audio.cpu()
        output_list = []
        for idx, (start, end) in enumerate(norm_list):
            split_utt = utt+'_'+str(idx)
            fpath = os.path.join(save_dir, split_utt+'.flac')
            split_start = (int)(start * self.fs)
            split_end = (int)(end * self.fs)
            splitted = (audio[split_start:split_end]*32767).short()
            torchaudio.save(fpath, splitted.unsqueeze(0), self.fs, bits_per_sample=16)
            output_list += [split_utt+'\t'+fpath+'\t'+audio_path+'\t'+str(start)+'\t'+str(end)+'\t'+str(end-start)+'\n']
        
        save_map_path = os.path.join(save_dir, utt+'.map')
        with open(save_map_path, 'w', encoding='utf-8') as fp:
            for line in output_list:
                fp.write(line)
        
        return save_map_path

    def forward(self, utt, wav, audio_path):
        # get function
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = self.utils
        
        print("########### ", utt, audio_path)

        # normalize to +/- 1.0
        wav = wav/wav.abs().max()
        maxTime = len(wav)/self.fs

        vad_iterator = VADIterator(self.model)

        tss = []
        window_size_samples = 1536 # number of samples in a single audio chunk
        total_frames = len(wav)//window_size_samples

        start_time = time.time()

        frame_idx = 0
        for i in range(0, len(wav), window_size_samples):
            chunk = wav[i: i+window_size_samples]
            if len(chunk) < window_size_samples:
                break
            
            with torch.no_grad():
                speech_dict = vad_iterator(chunk, return_seconds=True)
            
            if speech_dict:
                tss += [speech_dict]
            
            frame_idx += 1
            if frame_idx % 2000 == 0:
                print(f"frame is {frame_idx}/{total_frames} = {100*frame_idx/total_frames:.3f}%")

        vad_iterator.reset_states() # reset model states after each audio

        print(f'Total time of VAD is {time.time()-start_time}')

        if len(tss) >= 1:
            # merge vad intervals and normalize
            if 'end' not in tss[-1].keys():
                tss += [{'end':maxTime}]
            
            norm_list = self.timestamp_normalize(tss)

            # format output dir
            # dir_list = os.path.dirname(audio_path).split('/')[-2:]
            dir_list = os.path.dirname(audio_path).split('/')[-1:]
            dir_name = '/'.join(dir_list)

            segment_subdir = os.path.join(self.segments_dir, dir_name)
            audio_subdir = os.path.join(segment_subdir, utt)
            if not os.path.exists(audio_subdir):
                os.makedirs(audio_subdir)
            
            tg_path = os.path.join(audio_subdir, utt+'.TextGrid')
            self.generate_textgrid(norm_list, tg_path, maxTime)

            # split audio
            save_map_path = self.split_audio(wav, norm_list, utt, audio_path, audio_subdir)
            print("split_audio: done")

            processed = dir_name+'/'+utt
        else:
            processed = None
            save_map_path = None
            print('**************** ERROR', audio_path)

        return processed, save_map_path
    
    def forward_file(self, audio_path):
        # get function
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = self.utils
        
        # load audio
        wav = read_audio(audio_path, sampling_rate=self.fs)
        wav = wav.to('cuda')

        utt = os.path.basename(audio_path).split('.')[0]
        
        # get segments
        processed, save_map_path = self.forward(utt, wav, audio_path)

        return processed, save_map_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio_path", type=str, default="./test.flac", help="input audio file path")
    parser.add_argument("-o", "--output", type=str, default="./align_output", help="output directory")

    args = parser.parse_args()
    
    vad_obj = VAD(args)
    wav_path = "/mnt/cfs/SPEECH/zhangxinke1/work/audio/resources/audios/17_926361.wav"
    # wav_path = "/mnt/cfs/SPEECH/data/tts/tal_lessons/audio/8/295518.mp3"
    
    processed, save_map_path = vad_obj.forward_file(wav_path)
    print(f"Audio = {processed}, Map = {save_map_path}, Done!")
