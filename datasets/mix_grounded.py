from torch.utils.data import Dataset
import numpy as np
import torch
import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from mm_utils.video_utils import read_frames_decord, read_frames_av
from datasets.chat.base_template import LLaMA3_Template, Vicuna_Template, Phi_3_5_Template, DEFAULT_IMAGE_TOKEN, GROUNDING_TOKEN

class MixGrounded(Dataset):
    def __init__(
        self,
        anno_path = "data_path/mix_grounded/mix_grounded.json",
        video_path = "data_path",
        num_frames = 96,
        num_segs = 12,
        num_temporal_tokens = 300,
        sample='rand',
        llm='llama3',
    ):
        self.video_path = video_path
        self.num_frames = num_frames
        self.num_segs = num_segs
        self.num_temporal_tokens = num_temporal_tokens
        self.sample = sample

        self.data = load_json(anno_path)

        if llm == 'llama3':
            self.chat_template = LLaMA3_Template()
        elif llm == 'vicuna':
            self.chat_template = Vicuna_Template()
        elif llm == 'phi3.5':
            self.chat_template = Phi_3_5_Template()

        self.video_processor = frame_transform(image_size=224, mean=INTERNVIDEO_MEAN, std=INTERNVIDEO_STD)
        self.image_processor = frame_transform(image_size=336, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)

        self.video_ids = []
        self.question_ids = []
        self.video_files = []
        self.text_inputs = []
        self.dataset_names = []

        for item in self.data:
            self.question_ids.append(item['question_id'])
            self.video_files.append(item['video_file'])
            self.video_ids.append(item['video_id'])
            conversations = item['conversation']
            conversations = self.detect_timestamp_response(conversations)
            self.text_inputs.append(self.chat_template.encode(conversations))
            self.dataset_names.append(item['dataset_name'])

    def detect_timestamp_response(self, convs):
        pattern = r'<-?\d+(\.\d+)?>'
        for i in range(len(convs)):
            if i%2 == 0:
                if bool(re.search(pattern, convs[i+1]['value'])):
                    if DEFAULT_IMAGE_TOKEN in convs[i]['value']:
                        convs[i]['value'] = DEFAULT_IMAGE_TOKEN + ' ' + GROUNDING_TOKEN + '\n' + convs[i]['value'].replace(DEFAULT_IMAGE_TOKEN+'\n', '')
                    else:
                        convs[i]['value'] = GROUNDING_TOKEN + '\n' + convs[i]['value']
            else:
                continue
        return convs
    
    def convert_time_position(self, answer, duration):
        def replace_float(match):
            full_match = match.group(0)
            time = float(full_match.strip('<>'))
            quantized_time = int(self.num_temporal_tokens * time / duration)
            quantized_time = min(quantized_time, self.num_temporal_tokens)
            return f'<{quantized_time}>'
        pattern = r'<-?\d+(\.\d+)?>'
        # 替换匹配到的浮点数时间戳
        new_answer = re.sub(pattern, replace_float, answer)
        return new_answer

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        video_id = str(self.video_ids[index])
        question_id = str(self.question_ids[index])
        text_input = self.text_inputs[index]
        video_file = str(self.video_files[index])
        dataset_name = self.dataset_names[index]

        video_path = os.path.join(self.video_path, video_file)

        try:
            pixel_values, frame_indices, fps, total_frame_num, duration = read_frames_decord(
                video_path = video_path,
                num_frames = self.num_frames,
                sample = self.sample,
            )
        except Exception:
            print(f"read_frames_decord ERROR: {dataset_name}, {question_id}, {video_id}, {video_file}, {text_input}")
            try:
                pixel_values, frame_indices, fps, total_frame_num, duration = read_frames_av(
                    video_path = video_path,
                    num_frames = self.num_frames,
                    sample = self.sample,
                )
            except Exception:
                print(f"read_frames_av ERROR: {dataset_name}, {question_id}, {video_id}, {video_file}, {text_input}")
                pixel_values, frame_indices, fps, total_frame_num, duration = read_frames_decord(
                    video_path = './experiments/video0.mp4',
                    num_frames = self.num_frames,
                    sample = self.sample,
                )
                conversations = [
                    {"from": "human", "value": DEFAULT_IMAGE_TOKEN+'\n'+"Provide an overview of what happens."},
                    {"from": "gpt", "value": "A man silently narrates his experience driving an audi."}
                ]
                text_input = self.chat_template.encode(conversations)

        temporal_pixel_values = []
        for i in range(pixel_values.shape[0]): 
            temporal_pixel_values.append(self.video_processor(pixel_values[i]))
        temporal_pixel_values = torch.tensor(np.array(temporal_pixel_values)) # [num_frames, 3, 224, 224]

        num_frames_per_seg = int(self.num_frames // self.num_segs)
        indices_spatial = [(i*num_frames_per_seg) + int(num_frames_per_seg/2) for i in range(self.num_segs)]
        spatial_pixel_values = []
        for i_spatial in indices_spatial:
            spatial_pixel_values.append(self.image_processor(pixel_values[i_spatial]))
        spatial_pixel_values = torch.tensor(np.array(spatial_pixel_values)) # [num_segs, 3, 336, 336]


        return {
                "video_ids": video_id,
                "question_ids": question_id,
                "text_inputs": self.convert_time_position(text_input, duration),
                "temporal_pixel_values": temporal_pixel_values,
                "spatial_pixel_values": spatial_pixel_values,
                "dataset_names": dataset_name,
                "durations": float(duration),

            }

