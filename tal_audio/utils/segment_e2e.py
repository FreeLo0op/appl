import os
from selectors import EpollSelector
import time
import textgrid
import argparse
from pydub import AudioSegment

def segmentation(args):
    start_time = time.time()
    if not os.path.exists(args.segments_dir):
        os.makedirs(args.segments_dir)
    audio_raw_dict = {}
    with open(args.audio_map, "r") as ap:
        for line in ap.readlines():
            line_list = line.strip().split("\t")
            #print(line_list)
            audio_raw_dict[line_list[0]]=line_list[1]
    
    if args.mode == 'normal':
        with open(os.path.join(args.segments_dir, "segments.map"), "w") as segments_map:
            flag = 0
            cnt_dict = dict()
            with open(args.textgrid_map, "r") as tp:
                for line in tp.readlines():
                    line = line.strip().split("\t")
                    utt_name = line[0]
                    if utt_name == "NULL":
                        pass
                    else:
                        cut_num = 0
                        if utt_name in audio_raw_dict.keys():
                            wav_path=audio_raw_dict[utt_name]   # 原始音频
                        else:
                            continue
                        # wav_path = line[1]    # 降噪后音频

                        tg_path = line[1]
                        tg = textgrid.TextGrid()
                        tg.read(tg_path)
                        ali_dict = {}
                        ali_flag = 0
                        for word in tg.getList("line")[0]:
                            # print('---------->',word.mark, word.minTime, word.maxTime)
                            ali_dict.update({ali_flag:[word.mark, word.minTime, word.maxTime]})
                            ali_flag += 1
                        #print(ali_dict)
                        
                        #print ("ali_dict_1:", ali_dict_1)
                        ali_dict2 = {}
                        ali_flag2 = 0
                        for k, v in ali_dict.items():
                            if len(v[0]) > 0:
                                ali_dict2.update({ali_flag2:v})
                                ali_flag2 += 1
                
                        split_list = [0]
                        split_text = []
                        tmp_text = ''
                        for k, v in ali_dict2.items():
                            assert len(v[0]) > 0
                            if v[2] - v[1] > 1.5 and k != ali_flag2-1:
                                
                                # 在长于1.5s的token处切分
                                if k == 0:
                                    # print('------->',v)
                                    split_list[0] = (v[2]+v[1])/2 ####
                                    # split_list[0] = v[2]
                                    if args.languages in ['cn','ko','ja']:
                                        tmp_text += v[0]
                                    else:
                                        tmp_text += ' ' + v[0]
                                else:
                                    if len(tmp_text) > 1:
                                        split_list.append((v[2]+v[1])/2)
                                        ####
                                        # print((v[2]+v[1])/2)
                                        # split_list.append(v[2])
                                        if args.languages in ['cn','ko','ja']:
                                            tmp_text += v[0]
                                        else:
                                            tmp_text += ' ' + v[0]
                                        split_text.append(tmp_text)
                                        tmp_text = ''
                                        # tmp_text = v[0]
                                        
                                    else:
                                        # print('------------------------>',v)
                                        if args.languages in ['cn','ko','ja']:
                                            tmp_text += v[0]
                                        else:
                                            tmp_text += ' ' + v[0]
                                    # print()
                            elif k == ali_flag2-1:
                                if args.languages in ['cn','ko','ja']:
                                    tmp_text += v[0]
                                else:
                                    tmp_text += ' ' + v[0]
                                split_text.append(tmp_text)
                                # print('----------------->',tmp_text,split_text)
                            else:
                                if args.languages in ['cn','ko','ja']:
                                    tmp_text += v[0]
                                else:
                                    tmp_text += ' ' + v[0]


                        audio = AudioSegment.from_file(wav_path, format="flac")
                        split_list.append(len(audio)/1000)
                        tmp_wav = AudioSegment.empty()
                        tmp_len = 0
                        tmp_t = []

                        num = 1
                        for i in range(len(split_list)-1):
                            start, end = split_list[i]*1000, split_list[i+1]*1000
                            
                            # print(start,'\t###\t',end)

                            cut_audio = audio[start:end]
                            
                            if tmp_len + (split_list[i+1]-split_list[i]) < args.min_len:
                                tmp_wav += cut_audio
                                tmp_len += (split_list[i+1]-split_list[i])
                                tmp_t.append(split_text[i])
                            else:
                                cut_utt = utt_name + "_" + str(num).zfill(6)
                                cut_path = os.path.join(args.segments_dir, cut_utt + ".wav")
                                cut_path = os.path.abspath(cut_path)
                                tmp_t.append(split_text[i])
                                #print('output', tmp_v)
                                if args.languages not in ['cn','ko','ja']:
                                    new_text = ''.join(tmp_t)
                                    new_text = new_text.replace(' ','')
                                else:
                                    new_text = ' '.join(tmp_t)
                                    new_text = new_text.strip()
                                tmp_wav += cut_audio

                                tmp_wav = tmp_wav.set_channels(1)
                                tmp_wav = tmp_wav.set_sample_width(2)
                                tmp_wav = tmp_wav.set_frame_rate(16000)
                                tmp_wav.export(cut_path, format="wav")
                                cut_num += 1
                                num += 1
                                segments_map.write(cut_utt + "\t" + cut_path + "\t" + new_text + "\t" + '{:.2f}'.format(len(tmp_wav)/1000) + "\n")
                                #print(cut_utt + "\t" + cut_path + "\t" + new_text + "\t" + '{:.2f}'.format((end-start)/1000))
                                #segments_map.write(cut_utt + "\t" + cut_path + "\t" + new_text.replace(" ", " ") + "\t" + utt_name + "\n")
                                tmp_wav = AudioSegment.empty()
                                tmp_len = 0
                                tmp_t = []
                        
                        print('{} segmented into {} wavs'.format(utt_name, num-1))

    elif args.mode =='num':
            with open(os.path.join(args.segments_dir, "segments.map"), "w") as segments_map:
                cnt_dict = dict()
                with open(args.textgrid_map, "r") as tp:
                    for line in tp.readlines():
                        line = line.strip().split("\t")
                        utt_name = line[0]
                        if utt_name == "NULL":
                            pass
                        else:
                            cut_num = 0
                            if utt_name in audio_raw_dict.keys():
                                wav_path=audio_raw_dict[utt_name]  
                            else:
                                continue
                            
                            tg_path = line[2]
                            tg = textgrid.TextGrid()
                            tg.read(tg_path)
                            ali_dict_1 = {}
                            ali_flag = 0
                            for word in tg.getList("line")[0]:
                                ali_dict_1.update({ali_flag:[word.mark, word.minTime, word.maxTime]})
                                ali_flag += 1
                            
                            #print(ali_dict_1)
                            ali_dict2 = {}
                            ali_flag2 = 0
                            for k, v in ali_dict_1.items():
                                if len(v[0]) > 0:
                                    ali_dict2.update({ali_flag2:v})
                                    ali_flag2 += 1
                            
                            long_dict = {}
                            for k, v in ali_dict2.items():
                                if k not in [0, ali_flag2-1]:
                                    long_dict[k] = v[2]-v[1]
                            #print(sil_dict)
                            best_n_dict = {k:v for k, v in sorted(long_dict.items(), key=lambda x: x[1], reverse=True)[:args.cut_num-1]}
                            
                            time_stamps = [0]
                            for k in sorted(best_n_dict.keys()):
                                time_stamps.append((ali_dict2[k][2] + ali_dict2[k][1])/2)
                            
                            audio = AudioSegment.from_file(wav_path, format="flac")
                            for i in range(len(time_stamps)):
                                if i+1 < len(time_stamps):
                                    start = time_stamps[i] * 1000
                                    end = time_stamps[i+1] * 1000
                                else:
                                    start = time_stamps[i] * 1000
                                    end = len(audio)
                                
                                cut_audio = audio[start:end]
                                cut_utt = utt_name + "_" + str(i+1).zfill(6)
                                cut_path = os.path.join(args.segments_dir, cut_utt + ".wav")
                                cut_path = os.path.abspath(cut_path)
                                cut_audio = cut_audio.set_channels(1)
                                cut_audio = cut_audio.set_sample_width(2)
                                cut_audio = cut_audio.set_frame_rate(16000)
                                cut_audio.export(cut_path, format="wav")
                                segments_map.write(cut_utt + "\t" + cut_path + "\t" + utt_name + "\t" + '{:.2f}'.format((end-start)/1000) + "\n")
      
    end = time.time()
    running_time = end-start_time
    print ("Segmentation time cost: %.5f sec"%running_time)
                        
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--textgrid_map", type=str, default="./textgrids.map")
    parser.add_argument("-a", "--audio_map", type=str, default="./ted.map")
    parser.add_argument("-s", "--segments_dir", type=str, default="./ted30_segments2")
    parser.add_argument("-l", "--min_len", type=float, default=5.0)
    parser.add_argument("-m", "--mode", type=str, default='normal',help='normal,num')
    parser.add_argument("-n", "--cut_num", type=int, default=1,help='For mode num, ouput segment number')
    parser.add_argument("-lang", "--languages", type=str, default="en", help="ar, cn, yue, en, ko, id, ja, chuan, henan")
    args = parser.parse_args()
    segmentation(args)
