from model import SenseVoiceSmall
#from wenet.sense_voice.auto_model import AutoModel
#from wenet.sense_voice.postprocess_utils import rich_transcription_postprocess
import logging
from torch.utils.data import DataLoader
from longaudio_dataset import LongAudioDataset
import torch
from tqdm import tqdm
import yaml
#from wenet.utils.init_tokenizer import init_tokenizer
import copy
from sentencepiece_tokenizer import SentencepiecesTokenizer
import time
import argparse

sense_model_dir = "/mnt/cfs/SPEECH/dengtengyue1/wenet/examples/xpad/s0/premodel/SenseVoiceSmall"
#data_path = "/mnt/cfs/SPEECH/dengtengyue1/wenet/examples/xpad/s0/test/segments_01.json"
config = "/mnt/cfs/SPEECH/dengtengyue1/wenet/examples/xpad/s0/conf/train_sensevoice.yaml"
#output_path = "/mnt/cfs/SPEECH/dengtengyue1/wenet/examples/xpad/s0/result"
tokenizer_path = "/mnt/cfs/SPEECH/dengtengyue1/wenet/examples/xpad/s0/premodel/SenseVoiceSmall/chn_jpn_yue_eng_ko_spectok.bpe.model"
data_type = 'raw'
data_path2 = "/mnt/cfs/SPEECH/dengtengyue1/wespeaker/runtime/server/x86_gpu/data/zijian/enroll/speaker7/00001.wav"

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
parser = argparse.ArgumentParser(description='eval your network')
parser.add_argument('--input', 
                    default="/mnt/cfs/SPEECH/dengtengyue1/wenet/examples/xpad/s0/test/data4.json", 
                    help='input file')
parser.add_argument('--output', 
                    default="/mnt/cfs/SPEECH/dengtengyue1/wenet/examples/xpad/s0/pack/result", 
                    help='output file')
args = parser.parse_args()
data_path = args.input
output_path = args.output

#读取yaml文件等各参数
with open(config, 'r') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)
device = 'cuda'
test_conf = copy.deepcopy(configs['dataset_conf'])

#加载data和dataloader和tokenizer
#tokenizer = init_tokenizer(configs)
tokenizer = SentencepiecesTokenizer(bpemodel=tokenizer_path)
test_dataset = LongAudioDataset(data_type,
                       data_path,
                       tokenizer,
                       test_conf,
                       partition=False)

test_data_loader = DataLoader(test_dataset,
                              batch_size=None,
                              num_workers=0)

model, kwargs = SenseVoiceSmall.from_pretrained(sense_model_dir)
# emb = model.embed
# torch.save(emb.state_dict(),'/mnt/cfs/SPEECH/dengtengyue1/wenet/examples/xpad/s0/premodel/embed.pt')
######################################################################################
# print(kwargs)
# res = model.inference(
#     data_in=data_path2,
#     language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=False,
#     **kwargs,
# )
# print(res)
######################################################################################

model.to(device)
dtype = torch.float32
logging.info("compute dtype is {}".format(dtype))
model.eval()
print(model)
results = []
start_time = time.time()
with torch.cuda.amp.autocast(enabled=True,
                             dtype=dtype,
                             cache_enabled=False):
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_data_loader)):
            speech = batch['feats'].to(device)
            speech_lengths = batch['feats_lengths'].to(device)
            key = batch['keys']
            #tokens = batch['tokens']
            encoder_out, encoder_out_lens = model.encoder(speech, speech_lengths)
            # print(f"2 speech.shape: {speech.shape}")
            # print(f"2 speech_lengths: {speech_lengths}")
            # print(f"2 encoder_out.shape: {encoder_out.shape}")
            # print(f"2 encoder_out_lens: {encoder_out_lens}")
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
            ctc_logits = model.ctc.log_softmax(encoder_out)
            # print(f"2 ctc_logits.shape: {ctc_logits.shape}")

            b, n, d = encoder_out.size()
            if isinstance(key[0], (list, tuple)):
                key = key[0]
            if len(key) < b:
                key = key * b
            for i in range(b):
                x = ctc_logits[i, : encoder_out_lens[i].item(), :]
                yseq = x.argmax(dim=-1)#获取最大索引
                #print(f"2 最大索引序列: {yseq}")
                yseq = torch.unique_consecutive(yseq, dim=-1)#移除连续重复元素
                #print(f"2 最大索引序列移除重复元素: {yseq}")
                ibest_writer = None
                if kwargs.get("output_dir") is not None:
                    if not hasattr(self, "writer"):
                        self.writer = DatadirWriter(kwargs.get("output_dir"))
                    ibest_writer = self.writer[f"1best_recog"]

                mask = yseq != model.blank_id#生成掩码以过滤重复符号
                token_int = yseq[mask].tolist()

                # Change integer-ids to tokens
                # print(f"2 mask: {mask}")
                # print(f"2 生成的token序列: {token_int}")
                # print(f"2 实际的token: {tokens[i]}")
                
                text = tokenizer.tokens2text(token_int)

                result_i = {"key": key[i], "text": text}
                #print(result_i)
                results.append(result_i)

                if ibest_writer is not None:
                    ibest_writer["text"][key[i]] = text
        end_time = time.time()
        with open(output_path,'w') as f:
            for result in results:
                f.write(str(result) + '\n')
            #print(results)
print("结束")
total_inference_time_for_1000 = end_time - start_time
print(f"Total time for 1000 inferences: {total_inference_time_for_1000:.6f} seconds")






# print(kwargs)
# res = model.inference(
#     data_in="data",
#     language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=False,
#     **kwargs,
# )
# print("模型加载成功！")
# print(res)

# model = AutoModel(
#     model=sense_model_dir,
#     trust_remote_code=True,
# )
# print("模型加载成功！")
# res = model.generate(
#     input=data,
#     cache={},
#     language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=False,
# )

# text = rich_transcription_postprocess(res[0]["text"])

# print(text)