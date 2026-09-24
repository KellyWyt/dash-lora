import json
import ast
import os
from os.path import join,exists
import numpy as np
import pandas as pd
import cv2,csv
from typing import Sequence,Dict
from dataclasses import dataclass
import librosa
from PIL import Image
import torch
import random
import transformers
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from decord import VideoReader
from transformers import CLIPImageProcessor
import warnings

warnings.filterwarnings("ignore")
from dataset.audio_processor import preprocess



class UnifiedDataset(Dataset):
    def __init__(
        self,
        mode='train', # train,val,test
        video_processor: CLIPImageProcessor = None,
        tokenizer: PreTrainedTokenizer = None,
        image_size = 224,
        video_frame_nums = 10,
        avqa_task=False,
        ave_task = False
    ) -> None:
        super().__init__()

        self.mode=mode
        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.video_frame_nums = video_frame_nums

        self.samples = []
        self.tot = 0

        ### avqa data
        if avqa_task:
            self.add_avqa_task_samples()
        
        ## ave data
        if ave_task:
            self.add_ave_task_samples()
        
        print(f'tot training sample nums: {self.tot}')


    def add_avqa_task_samples(self):
        avqa_annotation_path = 'label_file/label_new/music-avqa-valid_train_samples.json'   # NOTE: change here for load information
        avqa_data_root = 'label_file/caption_gemini/avqa_converted_label'
        tot = 0
        my_path='/nfs1/outdated/WYT/data'   # NOTE: change here for load data
        with open(avqa_annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_id = sample['video_id']
            question_id = sample['question_id']
            _type = sample['type']
            video_path = my_path+sample['video_path'][24:]
            audio_path = my_path+sample['audio_path'][24:]
            question = sample['question']
            answer = sample['answer']
            # label_path = join(avqa_data_root,'converted_label',str(question_id)+'.txt')
            label_path = join(avqa_data_root, str(question_id)+'.txt')
            output = self.read_label(label_path)

            
            output_new=output


            instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\n<question_start>Please answer this question: {question}'+'<question_end>'
            self.samples.append(
                {
                    'vid':video_id,
                    'qid':question_id,
                    'type':_type,
                    'video_path':video_path,
                    'audio_path':audio_path,
                    # 'question':question,
                    # 'label_path':label_path,
                    'output': output_new,
                    # 'output':simple_output,
                    'task_name':'avqa',
                    'instruction':instruction,
                }
            )
            tot += 1
        print(f'avqa sample nums: {tot}')
        self.tot += tot


    def add_ave_task_samples(self):
        # NOTE modified for ave: 使用预处理好的视频帧和音频
        # Keep one training row per converted label.
        self.ave_annotation_path = 'MokA_AudioVisualText/AVE_data/train_samples_ave_deduplicated.json'
        ave_data_root = 'MokA_AudioVisualText/converted_label_1/converted_label'
        self.ave_preprocess_root = 'AVE_Dataset'  # 预处理后的数据根目录
        tot = 0

        if not os.path.exists(self.ave_annotation_path):
            print(f"Warning: {self.ave_annotation_path} not found.")
            return
        
        with open(self.ave_annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            event = sample['event']
            vid = sample['vid']
            start_time = sample['start_time']
            end_time = sample['end_time']

            # 构建在该模式下的具体路径
            # 视频路径示例: AVE_Dataset/train/video/_QQP43H56TA/
            # 音频路径示例: AVE_Dataset/train/audio/_QQP43H56TA.wav
            curr_video_path = join(self.ave_preprocess_root, self.mode, 'video', vid)
            curr_audio_path = join(self.ave_preprocess_root, self.mode, 'audio', vid + '.wav')
            label_path = join(ave_data_root,str(vid)+'.txt')
            output = self.read_label(label_path)
            output_new=output

            # 校验文件是否存在，防止训练中断
            if not exists(curr_audio_path):
                continue
            
            # instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nPlease describe the events and time range that occurred in the video.'
            instruction = (
            f'This is a video:\n<video_start><video><video_end>\n'
            f'This is an audio:\n<audio_start><audio><audio_end>\n'
            f'<question_start>Please describe the events and time range that occurred in the video.<question_end>'
            )
            # output = f'event:{event} start_time:{start_time} end_time:{end_time}'
            self.samples.append(
                {
                    'vid': vid,
                    'video_path': curr_video_path, # 存入具体路径
                    'audio_path': curr_audio_path,
                    'event': event,
                    'start_time': start_time,
                    'end_time': end_time,
                    'task_name':'ave',
                    'instruction':instruction,
                    'output': output_new,
                }
            )
            tot += 1
        print(f'ave sample nums: {tot}')
        self.tot += tot

    def read_label(self,label_path):
        with open(label_path,'r') as f:
            label = f.read()
        return label


    def __len__(self):
        return len(self.samples)


    def __getitem__(self,idx):

        sample = self.samples[idx]
        task_name = sample['task_name']
        instruction = sample['instruction']
        output = sample.get('output',None)
        # NOTE modified for ave: 如果 output 不存在且不是 ave 任务，才从 label_path 读取
        if output is None and task_name != 'ave':
            label_path = sample['label_path']
            output = self.read_label(label_path)
        if self.tokenizer is not None and hasattr(self.tokenizer,'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction},
            ]
            instruction = self.tokenizer.apply_chat_template(conversation=messages,add_generation_prompt=True,tokenize=False)
            # instruction = self.tokenizer.apply_chat_template(conversation=messages,add_generation_prompt=True,enable_thinking=False,tokenize=False) #NOTE qwen3

            # output = output + '<|eot_id|>'
            # output = output + '</s>'
            # # NOTE: 跑之前记得根据这里修改对应的终止符
            # output = output + '</s>'    # for Llama series
            output = output + ''    # for qwen2 series
        data = {
            'instruction':instruction,
            'output':output,
            'task_name':task_name,
        }
        
        if task_name == 'avqa':
            audio_path = sample['audio_path']
            video_path = sample['video_path']
            ### process video
            vr = VideoReader(uri=video_path, height=self.image_size, width=self.image_size)
            vlen = len(vr)
            start, end = 0, vlen
            n_frms = self.video_frame_nums
            n_frms = min(n_frms, vlen)
            indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
            # get_batch -> T, H, W, C
            temp_frms = vr.get_batch(indices).asnumpy()
            frames = []
            T = temp_frms.shape[0]
            for i in range(T):
                frame = Image.fromarray(temp_frms[i])
                frames.append(frame)
            frames = self.video_processor.preprocess(frames,return_tensors='pt')
            video = frames['pixel_values']  # t,c,h,w
            data['video'] = video
            
            ### process audio
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True) # NOTE 采样率为16000 Hz
            length = len(audio)# NOTE 音频总长度（采样点数）
            tot = 60 # NOTE 音频总时长60s
            nums_per_second = int(length / tot)# NOTE 每秒的采样点数
            indices = [i for i in range(0,60,6)] # NOTE 每6秒取一个中心点，共10个片段
            for indice in indices:
                start_time = max(0, indice - 0.5) # 中心点前0.5秒，中心点后1.5秒
                end_time = min(tot, indice + 1.5) # NOTE 每个片段时长2秒，（32000个采样点）
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                if indice - 0.5 < 0: # 左边界填充（第一个片段）
                    sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((sil, audio_seg),axis=0)
                if indice + 1.5 > tot: # 右边界填充（最后一个片段）  
                    sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0) # 2s音频作为输入 （1.32000）
                fbank = preprocess(audio_seg) # 提取特征
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens 这里L = 198，代表每个片段的时间帧数，每个L对应一个时间帧，包含了该时间点的频谱信息。
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128 list转tensor，更适合批量处理
            data['audio'] = audio_feature

        elif task_name == 'ave':
            # NOTE modified for ave: 从预处理好的视频帧和音频文件中读取
            vid = sample['vid']
            video_frame_folder = sample['video_path']
            audio_path = sample['audio_path']
            
            # 从预处理好的视频帧文件夹读取
            # video_frame_folder = join(self.ave_preprocess_root, self.mode, 'video', vid)
            frames = []
            for i in range(1, self.video_frame_nums + 1):
                frame_path = join(video_frame_folder, f'frame_{i}.jpg')
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            frames = self.video_processor.preprocess(frames,return_tensors='pt')
            video = frames['pixel_values']  # t,c,h,w
            data['video'] = video
            
            # 从预处理好的音频文件读取
            # audio_path = join(self.ave_preprocess_root, self.mode, 'audio', vid + '.wav')
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            length = len(audio)
            tot = 10  # 预处理后的音频时长为10秒
            nums_per_second = int(length / tot)
            
            # 每2秒取一个片段，共5个片段
            audio_feature = []
            indices = [i for i in range(0, 10, 2)]  # 0, 2, 4, 6, 8
            for indice in indices:
                start_time = max(0, indice)
                end_time = min(tot, indice + 2)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                if len(audio_seg) < 2 * nums_per_second:
                    sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32)
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature, dim=0)  # t, L, 128
            data['audio'] = audio_feature

        return data


class UnifiedTestDataset(Dataset):
    def __init__(
        self,
        mode='test', # train,val,test
        video_processor: CLIPImageProcessor = None,
        tokenizer: PreTrainedTokenizer = None,
        image_size = 224,
        video_frame_nums = 10,
        # avqa
        avqa_task=False,
        # ave task
        ave_task = False,
    ) -> None:
        super().__init__()

        self.mode=mode
        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.video_frame_nums = video_frame_nums



        self.samples = []
        self.tot = 0

        ### avqa data
        if avqa_task:
            self.add_avqa_task_samples()

        ### ave data
        if ave_task:
            self.add_ave_task_samples()

        
        print(f'tot test sample nums: {self.tot}')


    def add_avqa_task_samples(self):
        avqa_annotation_path = 'label_file/label_new/music_avqa_test_samples.json' #label_file/label_new/music_avqa_test_samples.json
        avqa_data_root = '/NFS1/WYT/MUSIC-AVQA/data/' #/group/40061/cserdu/data/music-avqa
        tot = 0
        my_path='/nfs1/outdated/WYT/data'   # replace this for load test data
        with open(avqa_annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_id = sample['video_id']
            question_id = sample['question_id']
            questio_type = sample['type']
            video_path = my_path+sample['video_path'][24:]
            audio_path = my_path+sample['audio_path'][24:]
            question = sample['question']
            answer = sample['answer']
            instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\n<question_start>Please answer this question: {question}'+'<question_end>'
            self.samples.append(
                {
                    'vid':video_id,
                    'qid':question_id,
                    'question_type':questio_type,
                    'video_path':video_path,
                    'audio_path':audio_path,
                    'question':question,
                    'task_name':'avqa',
                    'instruction':instruction,
                    'output': answer,
                }
            )
            tot += 1
        print(f'avqa sample nums: {tot}')


    def add_ave_task_samples(self):
        # NOTE modified for ave: 使用预处理好的视频帧和音频
        self.ave_annotation_path = 'MokA_AudioVisualText/AVE_data/test_samples_ave.json'
        self.ave_data_root = 'AVE_Dataset'
        self.ave_preprocess_root = 'AVE_Dataset'  # 预处理后的  数据根目录
        tot = 0
        with open(self.ave_annotation_path,'r') as f:
            samples = json.load(f)
        for sample in samples:
            event = sample['event']
            vid = sample['vid']
            # 重要：同步训练集的路径构建逻辑
            curr_video_path = join(self.ave_preprocess_root, self.mode, 'video', vid)
            curr_audio_path = join(self.ave_preprocess_root, self.mode, 'audio', vid + '.wav')
            start_time = sample['start_time']
            end_time = sample['end_time']
            # instruction = f'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\n<question_start>Please describe the events and time range that occurred in the video.' + <question_end>'
            instruction = (
            f'This is a video:\n<video_start><video><video_end>\n'
            f'This is an audio:\n<audio_start><audio><audio_end>\n'
            f'<question_start>Please describe the events and time range that occurred in the video.<question_end>'
            )
            # NOTE modified for ave: 直接设置 output
            output = f'event:{event} start_time:{start_time} end_time:{end_time}'
            self.samples.append(
                {
                    'vid': vid,
                    'video_path': curr_video_path, # 存入路径供 __getitem__ 使用
                    'audio_path': curr_audio_path,
                    'event': event,
                    'start_time': start_time,
                    'end_time': end_time,
                    'task_name':'ave',
                    'instruction':instruction,
                    'output': output,  # NOTE modified for ave: 直接设置 output
                }
            )
            tot += 1
        print(f'ave sample nums: {tot}')
        self.tot += tot



    def __len__(self):
        return len(self.samples)


    def read_label(self,label_path):
        if not os.path.exists(label_path):
            return 'no label.'
        with open(label_path,'r') as f:
            label = f.read()
        return label


    def __getitem__(self,idx):
        sample = self.samples[idx]
        task_name = sample['task_name']
        instruction = sample['instruction']
        output = sample.get('output',None)
        # NOTE modified for ave: 如果 output 不存在且不是 ave 任务，才从 label_path 读取
        if output is None and task_name != 'ave':
            label_path = sample['label_path']
            output = self.read_label(label_path)
        if self.tokenizer is not None and hasattr(self.tokenizer,'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction},
            ]
            instruction = self.tokenizer.apply_chat_template(conversation=messages,add_generation_prompt=True,tokenize=False)
            # output = output + '<|eot_id|>'    # Llama3
            # output = output + '</s>'
            # # NOTE: 跑之前记得根据使用的模型修改对应的终止符
            # output = output + '</s>'    # for Llama2 series
            output = output + ''    # for Qwen2 series
        
        data = {
            'instruction': instruction,
            'output': output,
            'task_name':task_name,
        }
        
        if task_name=='avqa':
            audio_path = sample['audio_path']
            video_path = sample['video_path']
            ### process video
            vr = VideoReader(uri=video_path, height=self.image_size, width=self.image_size)
            vlen = len(vr)
            start, end = 0, vlen
            n_frms = self.video_frame_nums
            n_frms = min(n_frms, vlen)
            indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
            # get_batch -> T, H, W, C
            temp_frms = vr.get_batch(indices).asnumpy()
            frames = []
            T = temp_frms.shape[0]
            for i in range(T):
                frame = Image.fromarray(temp_frms[i])
                frames.append(frame)
            frames = self.video_processor.preprocess(frames,return_tensors='pt')
            video = frames['pixel_values']  # t,c,h,w
            data['video'] = video
            data['video_path'] = video_path
            
            ### process audio
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 60
            nums_per_second = int(length / tot)
            indices = [i for i in range(0,60,6)]
            for indice in indices:
                start_time = max(0, indice - 0.5)
                end_time = min(tot, indice + 1.5)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                if indice - 0.5 < 0:
                    sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((sil, audio_seg),axis=0)
                if indice + 1.5 > tot:
                    sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data['audio'] = audio_feature
            data['audio_path'] = audio_path

            question_type = sample['question_type']
            vid = sample['vid']
            qid = sample['qid']
            data['question_type'] = question_type
            data['vid'] = vid
            data['qid'] = qid

        elif task_name == 'ave':
            # NOTE modified for ave: 从预处理好的视频帧和音频文件中读取
            vid = sample['vid']
            
            # 从预处理好的视频帧文件夹读取

            video_frame_folder = sample.get('video_path', join(self.ave_preprocess_root, self.mode, 'video', vid))
            audio_path = sample.get('audio_path', join(self.ave_preprocess_root, self.mode, 'audio', vid + '.wav'))
            if not exists(video_frame_folder):
                raise FileNotFoundError(f"AVE video frame folder not found: {video_frame_folder}")
            if not exists(audio_path):
                raise FileNotFoundError(f"AVE audio file not found: {audio_path}")

            frames = []
            for i in range(1, self.video_frame_nums + 1):
                frame_path = join(video_frame_folder, f'frame_{i}.jpg')
                if not exists(frame_path):
                    raise FileNotFoundError(f"AVE frame not found: {frame_path}")
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            frames = self.video_processor.preprocess(frames,return_tensors='pt')
            video = frames['pixel_values']  # t,c,h,w
            data['video'] = video
            data['video_path'] = video_frame_folder

            
            # 从预处理好的音频文件读取
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            length = len(audio)
            tot = 10  # 预处理后的音频时长为10秒
            nums_per_second = int(length / tot)
            
            # 每2秒取一个片段，共5个片段
            audio_feature = []
            indices = [i for i in range(0, 10, 2)]  # 0, 2, 4, 6, 8
            for indice in indices:
                start_time = max(0, indice)
                end_time = min(tot, indice + 2)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                if len(audio_seg) < 2 * nums_per_second:
                    sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32)
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature, dim=0)  # t, L, 128
            data['audio'] = audio_feature
            data['audio_path'] = audio_path



        return data



@dataclass
class DataCollatorForUnifiedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer
        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_task_names = []

        for instance in instances:
            instruction=instance['instruction']
            output=instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)
            
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            input_ids = instruction_ids + output_ids
            label = [-100] * len(instruction_ids) + output_ids #instruction部分被掩码，只预测output部分
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            
            X_modals = {}
            image = instance.get('image',None)
            if image is not None:
                X_modals['<image>'] = image
                
            video = instance.get('video',None)
            if video is not None:
                X_modals['<video>'] = video

            audio = instance.get('audio',None)
            if audio is not None:
                X_modals['<audio>'] = audio

            
            batch_X_modals.append(X_modals)

        
        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_task_names':batch_task_names
        }


@dataclass
class DataCollatorForUnifiedTestDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer
        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_metadata=[]
        batch_task_names = []

        for instance in instances:
            instruction = instance['instruction']
            output = instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)

            metadata = {
                'instruction': instruction,
                'output': output,
            }
            
            if task_name == 'avqa':
                question_type = instance.get('question_type',None)
                vid = instance.get('vid',None)
                qid = instance.get('qid',None)
                metadata.update(
                    {
                        'question_type':question_type,
                        'vid':vid,
                        'qid':qid
                    }
                )
            
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            
            # if task_name in ['ms3','s4','avss','ref-avs']:
            #     input_ids = instruction_ids + output_ids
            #     label = [-100] * len(instruction_ids) + output_ids
            # else:
            #     input_ids = instruction_ids
            #     label = [-100] * len(instruction_ids)

            input_ids = instruction_ids
            label = [-100] * len(instruction_ids)
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            X_modals = {}
            image = instance.get('image',None)
            if image is not None:
                # print("load images")
                X_modals['<image>'] = image
                metadata['image_path'] = instance.get('image_path','')
                
            video = instance.get('video',None)
            if video is not None:
                # print("load videos")
                X_modals['<video>'] = video
                metadata['video_path'] = instance.get('video_path','')

            audio = instance.get('audio',None)
            if audio is not None:
                # print("load audios")
                X_modals['<audio>'] = audio
                metadata['audio_path'] = instance.get('audio_path','')
            
            batch_X_modals.append(X_modals)
            batch_metadata.append(metadata)

        
        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_metadata':batch_metadata,
            'batch_task_names':batch_task_names,
        }


@dataclass
class DataCollatorForMsAudioUnifiedTestDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer
        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_metadata=[]
        batch_task_names = []

        for instance in instances:
            instruction = instance['instruction']
            output = instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)

            metadata = {
                'instruction': instruction,
                'output': output,
            }
            
            if task_name == 'avqa':
                question_type = instance.get('question_type',None)
                vid = instance.get('vid',None)
                qid = instance.get('qid',None)
                metadata.update(
                    {
                        'question_type':question_type,
                        'vid':vid,
                        'qid':qid
                    }
                )
            
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            
            # if task_name in ['ms3','s4','avss','ref-avs']:
            #     input_ids = instruction_ids + output_ids
            #     label = [-100] * len(instruction_ids) + output_ids
            # else:
            #     input_ids = instruction_ids
            #     label = [-100] * len(instruction_ids)

            input_ids = instruction_ids
            label = [-100] * len(instruction_ids)
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            X_modals = {}
            image = instance.get('image',None)
            if image is not None:
                # print("load images")
                image = torch.ones_like(image)
                X_modals['<image>'] = image
                metadata['image_path'] = instance.get('image_path','')
                
            video = instance.get('video',None)
            if video is not None:
                # print("load videos")
                # print(f"videos shape: {video.shape}")
                # video = torch.ones_like(video)
                X_modals['<video>'] = video
                metadata['video_path'] = instance.get('video_path','')

            audio = instance.get('audio',None)
            if audio is not None:
                # print("load audios")
                # print(f"audios shape: {audio.shape}")
                audio = torch.ones_like(audio)
                X_modals['<audio>'] = audio
                metadata['audio_path'] = instance.get('audio_path','')
            
            batch_X_modals.append(X_modals)
            batch_metadata.append(metadata)

        
        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_metadata':batch_metadata,
            'batch_task_names':batch_task_names,
        }

@dataclass
class DataCollatorForMsVideoUnifiedTestDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer
        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_metadata=[]
        batch_task_names = []

        for instance in instances:
            instruction = instance['instruction']
            output = instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)

            metadata = {
                'instruction': instruction,
                'output': output,
            }
            
            if task_name == 'avqa':
                question_type = instance.get('question_type',None)
                vid = instance.get('vid',None)
                qid = instance.get('qid',None)
                metadata.update(
                    {
                        'question_type':question_type,
                        'vid':vid,
                        'qid':qid
                    }
                )
            
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            
            # if task_name in ['ms3','s4','avss','ref-avs']:
            #     input_ids = instruction_ids + output_ids
            #     label = [-100] * len(instruction_ids) + output_ids
            # else:
            #     input_ids = instruction_ids
            #     label = [-100] * len(instruction_ids)

            input_ids = instruction_ids
            label = [-100] * len(instruction_ids)
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            X_modals = {}
            image = instance.get('image',None)
            if image is not None:
                # print("load images")
                image = torch.ones_like(image)
                X_modals['<image>'] = image
                metadata['image_path'] = instance.get('image_path','')
                
            video = instance.get('video',None)
            if video is not None:
                # print("load videos")
                # print(f"videos shape: {video.shape}")
                video = torch.ones_like(video)
                X_modals['<video>'] = video
                metadata['video_path'] = instance.get('video_path','')

            audio = instance.get('audio',None)
            if audio is not None:
                # print("load audios")
                # print(f"audios shape: {audio.shape}")
                # audio = torch.ones_like(audio)
                X_modals['<audio>'] = audio
                metadata['audio_path'] = instance.get('audio_path','')
            
            batch_X_modals.append(X_modals)
            batch_metadata.append(metadata)

        
        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_metadata':batch_metadata,
            'batch_task_names':batch_task_names,
        }


def get_dataset_collator(
    data_args,tokenizer: transformers.PreTrainedTokenizer,
    image_processor=None,mode='train'):
    if mode == 'train':
        dataset = UnifiedDataset(
            video_processor=image_processor,
            tokenizer=tokenizer,
            avqa_task=data_args.avqa_task,
            ave_task=data_args.ave_task
        )
        data_collator = DataCollatorForUnifiedDataset(tokenizer=tokenizer)
    
    elif mode == 'test':
        dataset = UnifiedTestDataset(
            video_processor=image_processor,
            tokenizer=tokenizer,
            avqa_task=data_args.avqa_task,
            ave_task=data_args.ave_task
        )
        data_collator = DataCollatorForUnifiedTestDataset(tokenizer=tokenizer)

    elif mode == 'test_with_msAudio':
        dataset = UnifiedTestDataset(
            video_processor=image_processor,
            tokenizer=tokenizer,
            avqa_task=data_args.avqa_task,
            ave_task=data_args.ave_task
        )
        data_collator = DataCollatorForMsAudioUnifiedTestDataset(tokenizer=tokenizer)
    elif mode == 'test_with_msVideo':
        dataset = UnifiedTestDataset(
            video_processor=image_processor,
            tokenizer=tokenizer,
            avqa_task=data_args.avqa_task,
            ave_task=data_args.ave_task
        )
        data_collator = DataCollatorForMsVideoUnifiedTestDataset(tokenizer=tokenizer)
    
    return dataset,data_collator

