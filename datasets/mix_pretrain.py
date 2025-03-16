from torch.utils.data import Dataset
import random
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import pickle
import sys
import os
import requests
from collections import Counter
from io import BytesIO
import json
import cv2
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from mm_utils.video_utils import read_frames_decord, read_frames_av
from datasets.chat.base_template import LLaMA3_Template, Vicuna_Template, Phi_3_5_Template

class MixPretrain(Dataset):
    def __init__(
        self,
        anno_path = "/home/haibo/data/mix_pretrain/mix_pretrain.json",
        video_path = "/home/haibo/data",
        num_frames = 96,
        num_segs = 12,
        num_temporal_tokens = 300,
        sample='rand',
        llm='phi3.5',
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
            self.text_inputs.append(self.chat_template.encode(conversations))
            self.dataset_names.append(item['dataset_name'])

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
                    {"from": "human", "value": "<image>\n"+"Provide an overview of what happens."},
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
                "text_inputs": text_input,
                "temporal_pixel_values": temporal_pixel_values,
                "spatial_pixel_values": spatial_pixel_values,
                "dataset_names": dataset_name,
            }

# dataset = MixPretrain(llm='phi3.5')
# for i in range(10):
#     entry = random.choice(dataset)
#     print(entry['question_ids'], entry['video_ids'], entry['dataset_names'])
#     print("text_inputs: ",             entry['text_inputs'])
#     print("temporal_pixel_values: ",             entry['temporal_pixel_values'].shape)
#     print("spatial_pixel_values: ",             entry['spatial_pixel_values'].shape)
#     print()
# print(len(dataset))
