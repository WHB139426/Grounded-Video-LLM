import torch
from models.llava_next_video import LLAVA_NEXT_VIDEO


from huggingface_hub import HfApi
from huggingface_hub import login
import zipfile
import os
from tqdm import tqdm

# login()
# api = HfApi()

# dirs = ['Phi-3.5-vision-instruct', 'Phi-3.5-mini-instruct', 'internvideo', 'ckpt', 'Phi-3.5-vision-instruct-seperated']

# for dir in dirs:
#     api.upload_folder(
#         folder_path=f"/home/haibo/weight_path/{dir}",
#         path_in_repo=dir,
#         repo_id=f"WHB139426/Grounded-Video-LLM",
#         repo_type="model",
#     )



device = 'cuda:0'

"""
Model 
"""
dtype = torch.float16
stage = 'sft'
max_txt_len = 2048
num_frames = 96
num_segs = 12
num_temporal_tokens=300
lora=True
llm='phi3.5'
attn_implementation='eager'
config_path = "/home/haibo/weight_path/Phi-3.5-vision-instruct"
tokenizer_path = "/home/haibo/weight_path/Phi-3.5-mini-instruct"
pretrained_video_path = '/home/haibo/weight_path/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt'
pretrained_vision_proj_llm_path = '/home/haibo/weight_path/Phi-3.5-vision-instruct-seperated/'
ckpt_path='/home/haibo/weight_path/ckpt/sft_llava_next_video_phi3.5_mix_sft_multi_modal_projector_video_projecter_language_model.pth'


model = LLAVA_NEXT_VIDEO(
    dtype=dtype, 
    stage=stage, 
    max_txt_len=max_txt_len, 
    num_frames=num_frames,
    num_segs=num_segs,
    num_temporal_tokens=num_temporal_tokens,
    lora=lora,
    llm=llm,
    attn_implementation=attn_implementation,
    config_path=config_path,
    tokenizer_path=tokenizer_path,
    pretrained_video_path=pretrained_video_path,
    pretrained_vision_proj_llm_path=pretrained_vision_proj_llm_path, 
    )
ckpt = torch.load(ckpt_path, map_location='cpu')['model']
if 'multi_modal_projector' in ckpt.keys():
    model.multi_modal_projector.load_state_dict(ckpt['multi_modal_projector'])
if 'video_projecter' in ckpt.keys():
    model.video_projecter.load_state_dict(ckpt['video_projecter'])
if 'language_model' in ckpt.keys():
    model.language_model.load_state_dict(ckpt['language_model'])  


model.to(device)

