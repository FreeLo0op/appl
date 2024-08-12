import os
from funasr import AutoModel
from textgrid import TextGrid, IntervalTier


class ASR:
    def __init__(self, args) -> None:
        # ToDo: Move align to seperate module
        # if args.align:
        #     self.model = AutoModel(model="fa-zh", model_revision="v2.0.4")
        # else:

        self.model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                                # vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                                # punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                                # spk_model="cam++", spk_model_revision="v2.0.2",
                                )
        
        self.funasr_punc_list = ['，', '。', '？', '、']
        self.fs = 16000
    
    def generate_alignment(self, utts, texts, timestamps, audio_lens, opaths):
        for idx, utt in enumerate(utts):
            maxTime = audio_lens[idx]
            endTime = maxTime
            tg = TextGrid(maxTime=maxTime)
            
            tier_graphemes = IntervalTier(name="graphemes")
            for text, timestamp in zip(texts[idx], timestamps[idx]):
                minTime = timestamp[0]/1000. # secs
                maxTime = timestamp[1]/1000. # secs
                grapheme_interval = [minTime, maxTime, text]
                tier_graphemes.add(*grapheme_interval)

            if maxTime < endTime:
                grapheme_interval = [maxTime, endTime, ""]
                tier_graphemes.add(*grapheme_interval)
            
            tg.append(tier_graphemes)

            # get output dir
            save_path = opaths[idx]
            asr_subdir = os.path.dirname(save_path)
            if not os.path.exists(asr_subdir):
                os.makedirs(asr_subdir)
            
            tg_path = os.path.join(asr_subdir, utt+'.TextGrid')
            tg.write(tg_path)

            with open(os.path.join(asr_subdir, utt+'.txt'), 'w', encoding='utf-8') as fp:
                fp.write(' '.join(texts[idx])+'\n')

    # def forward(self, utt, audio, audio_path, hotword='小思小思'):
    def forward(self, utt, audio, opath, hotword='小思小思'):
        batch_size = len(utt)

        # used to do asr
        audio_list = []
        if not isinstance(audio, (list, tuple)):
            # tensor
            if audio.dim() > 1:
                audio_list = [x for x in audio]
            else:
                audio_list = [audio]
        else:
            audio_list = audio
        
        res = self.model.generate(input=audio_list, 
                                  batch_size=batch_size,
                                  disable_pbar=True,
                                  hotword=hotword,
                                  )

        # used to do alignment
        # res = self.model.generate(input=(audio.squeeze(dim=0), text), data_type=("sound", "text"))
        
        utts = [item for item in utt]
        opaths = [item for item in opath]
        # texts = [item["raw_text"].strip().split() for item in res] # no "raw_text" in new funasr release
        texts = [item["text"].strip().split() for item in res]
        timestamps = [item["timestamp"] for item in res]
        audio_lens = [audio_list[ii].shape[-1]/self.fs for ii in range(batch_size)]

        self.generate_alignment(utts, texts, timestamps, audio_lens, opaths)

        return [' '.join(item["text"].strip().split()) for item in res]