#%%
from funasr import AutoModel

model = AutoModel(model="damo/speech_rwkv_bat_asr-zh-cn-16k-aishell1-vocab4234-pytorch", 
                  model_revision="v2.0.2",
                 )


#%%
audio_path = "/mnt/cfs/SPEECH/data/tts/tal_lessons/splits/segments/11/900091/900091_43.flac"
res = model.generate(input=audio_path)

print(audio_path)
print(res)


# %%
print(model.model)

# %%
