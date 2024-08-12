# tal_audio

# 1.工具使用说明

## 1.1 特征提取工具
```
在shell中执行：bash extract.sh extract_type map_path

# For example, when extracting pitch, follow instructions as below:
# step 1:
bash extract.sh mkdirs map_path

# step 2:
bash extract.sh pitch map_path
```

参数列表：
```
extract_type: mkdirs, spk, w2v, pitch
map_path: 输入map，格式是5列，第一列是utt，第二列是音频路径，第三列是text，第四列是时长，第五列是输出路径
```

# 环境
```bash
source /mnt/cfs/SPEECH/hupeng/tools/others_env/hp_env.sh 
conda activate appl
```

# 新增项目
## embedding提取
输入：
```
1、数据
example: /mnt/cfs/SPEECH/hupeng/oworkdir/audio_book_process/selected_2wh_books/set1_single.list
有效字段： 第一列：utt 第三列：音频路径（相对路径）
2、根目录&保存 路径
```
使用
```py
python tal_audio/run_embedding.py data_list output_dir
```
输出：
```
pt文件 每个segment的起始、结束时间、embedding
```

## vad后处理
输入：
```
1、数据
example：/mnt/cfs/SPEECH/hupeng/oworkdir/audio_book_process/part_00/part_00.list
有效字段： 第一列：utt 第三列：音频路径（相对路径） 最后一列：duration
2、根目录
3、输出文件
```
使用
```py
python vad_post_process.py data_list root_dir fo_file
```

## asr
输入：
```
1、数据
example:/mnt/cfs/SPEECH/data/tts/Audio_Book/part_00/segments/segments_01.list
2、batch_size
```
使用
```bash
bash asr_run.sh data_list batch_size
```

## sensevoice
输入：
```
1、数据
example:/mnt/cfs/SPEECH/dengtengyue1/wenet/examples/xpad/s0/test/audio_book/mini_test/data4.json
{"key": "e9326e4bb84a622e_7", "wav": "/mnt/cfs/SPEECH/data/tts/Audio_Book/part_00/seperate_data/data/book_0000/e9326e4bb84a622e.mp3", "start": "60.59375", "end": "64.5625", "total": "3.96875"}

2、输出
```
使用
```
python tal_audio/sensevoice/sensevoice_test.py --input data_path --output output_path
```