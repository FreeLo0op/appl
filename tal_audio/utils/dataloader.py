import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import time
from tqdm import tqdm

class AudioDataset(Dataset):
    def __init__(self, 
                 annotations, 
                 sep: str='\t',
                 audio_load: bool=False,
                 sort_order: str='desc',#str='desc',
                 target_sample_rate: int=16000,
                 target_channel=1
                ):
        """
        Args:
            annotations_file (str): Path to the annotations file with tab-separated values.
            sep (str): The delimiter for the annotations file.
            audio_load (bool): Whether to load and process audio segments.
            sort_order (str): Sorting order for the segments based on their duration. Can be 'asc' or 'desc'.
        """
        if isinstance(annotations, pd.DataFrame):
            self.annotations = annotations
        elif isinstance(annotations, str):
            self.annotations = pd.read_csv(annotations, sep=sep)

        self.sort_order = sort_order
        if self.sort_order:
            self.annotations['duration'] = self.annotations['duration'].astype(float)
        #    self.annotations.sort_values(by='duration', ascending=(sort_order == 'asc'), inplace=True)
        self.audio_load = audio_load
        if self.audio_load:
            self.target_sample_rate = target_sample_rate
            self.audio_cache = {}
            self.audio_segment_counts = self._count_segments()

        self.target_channel = target_channel
        
    def _count_segments(self):
        """
        Counts the number of segments per audio file.
        """
        counts = self.annotations['audio'].value_counts().to_dict()
        return counts

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.annotations.iloc[idx]
        utt, audio_file, start_time, end_time, duration = row['utt'], row['audio'], row['start_time'], row['end_time'], row['duration']
        
        if self.audio_load:
            # Load the audio file if it's not already loaded
            if audio_file not in self.audio_cache:
                waveform, sample_rate = torchaudio.load(audio_file, normalize=True)
                self.audio_cache[audio_file] = (waveform, sample_rate)
            else:
                waveform, sample_rate = self.audio_cache[audio_file]
            # Convert start and end time to sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
        
            # Extract the audio segment
            #if waveform.numel() == 0 or start_sample >= waveform.size(1) or end_sample > waveform.size(1):
            #    print(f"Warning: Empty or invalid audio segment for {audio_file}")
            #    return utt, torch.tensor([]), audio_file
            audio_segment = waveform[:, start_sample:end_sample]
            if audio_segment.size(1) == 0:
                return None
            
            if audio_segment.size(0) > 1:
                audio_segment = audio_segment.mean(dim=0, keepdim=True)
            if audio_segment.size(0) >= self.target_channel:
                audio_segment = audio_segment[:self.target_channel, :]
            else:
                audio_list = [audio_segment[0:1, :] for _ in range(self.target_channel)]
                audio_segment = torch.concat(audio_list, dim=0)
            # Resample the audio segment to the target sample rate
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                audio_segment = resampler(audio_segment)
            
            # Decrement the segment count and remove from cache if all segments are processed
            self.audio_segment_counts[audio_file] -= 1
            if self.audio_segment_counts[audio_file] == 0:
                del self.audio_cache[audio_file]

            return utt, audio_segment.squeeze(0), audio_file, duration
        else:
            # Return the raw audio info without loading the actual audio file
            return utt, audio_file, start_time, end_time

def collate_fn(batch):
    batch = [item for item in batch if item is not None] 
    if isinstance(batch[0][1], torch.Tensor):        
        utts, audio_segments, audio_files, durations = zip(*batch)
        return utts, audio_segments, audio_files, durations
    else:
        utts, audio_files, start_times, end_times = zip(*batch)
        return utts, audio_files, start_times, end_times

if __name__ == '__main__':
    # Example usage:
    annotations_file = "/mnt/cfs/SPEECH/hupeng/audio_ppl/data_list/test_dataloader_2long.list"
    root_dir = r'/mnt/cfs/SPEECH/hupeng/audio_ppl/test_audios_4'

    # Create the dataset and dataloader for audio segments
    t1 = time.time()
    audio_segment_dataset = AudioDataset(annotations_file, audio_load=True, sort_order='desc')
    audio_segment_loader = DataLoader(audio_segment_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=8)
    for batch in tqdm(audio_segment_loader):
        utts, audio_segments, audio_files, durations = batch
        print(durations)
    t2 = time.time()
    print(t2-t1)
    sys.exit(0)

    # Create the dataset and dataloader for audio info only
    audio_info_dataset = AudioDataset(annotations_file, audio_load=False)
    audio_info_loader = DataLoader(audio_info_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Iterate through the data for audio segments
    print("Audio Segment Loader:")
    for batch in audio_segment_loader:
        utts, audio_segments, audio_files = batch
        for utt, audio, audio_file in zip(utts, audio_segments, audio_files):
            print(utt, audio.shape, audio_file)
            # Save file if needed
            #save_file = os.path.join(root_dir, f'{utt}.mp3')
            #torchaudio.save(save_file, audio, rate)

    # Iterate through the data for audio info
    print("Audio Info Loader:")
    for batch in audio_info_loader:
        utts, audio_files, start_times, end_times = batch
        for utt, audio_file, start_time, end_time in zip(utts, audio_files, start_times, end_times):
            print(utt, audio_file, start_time, end_time)
