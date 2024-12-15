import torch
import argparse
import numpy as np
import random
from torch.backends import cudnn
import re
from torch.cuda.amp import autocast as autocast
from datasets.chat.base_template import LLaMA3_Template, Vicuna_Template, Phi_3_5_Template, DEFAULT_IMAGE_TOKEN, GROUNDING_TOKEN
from models.llava_next_video import LLAVA_NEXT_VIDEO
from mm_utils.video_utils import read_frames_decord
from mm_utils.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dtype', type=torch.dtype, default=torch.bfloat16, choices=[torch.bfloat16, torch.float16, torch.float32]) # choose 'torch.float16' if your device doesn't support bfloat16

    # model
    parser.add_argument('--model', type=str, default='llava_next_video', choices=['llava_next_video'])
    parser.add_argument('--llm', type=str, default='phi3.5', choices=['llama3', 'vicuna', 'phi3.5'])
    parser.add_argument('--stage', type=str, default="sft", choices=['pretrain', 'grounded', 'sft'])
    parser.add_argument('--max_txt_len', type=int, default=2048)
    parser.add_argument('--num_temporal_tokens', type=int, default=300)
    parser.add_argument('--num_frames', type=int, default=96)
    parser.add_argument('--num_segs', type=int, default=12)
    parser.add_argument('--lora', type=bool, default=True)
    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2", choices=['eager', 'flash_attention_2']) # choose 'eager' if you cannot install flash_attention_2

    # path
    parser.add_argument('--config_path', type=str, default="weight_path/Phi-3.5-vision-instruct")
    parser.add_argument('--tokenizer_path', type=str, default="weight_path/Phi-3.5-mini-instruct")
    parser.add_argument('--pretrained_video_path', type=str, default='weight_path/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt')
    parser.add_argument('--pretrained_vision_proj_llm_path', type=str, default='weight_path/Phi-3.5-vision-instruct-seperated/')
    parser.add_argument('--ckpt_path', type=str, default='weight_path/ckpt/sft_llava_next_video_phi3.5_mix_sft_multi_modal_projector_video_projecter_language_model.pth')

    # inputs
    parser.add_argument('--prompt_grounding', type=str, default="Give you a textual query: 'The female host wearing purple clothes is reporting news in the studio'. When does the described content occur in the video? Please return the start and end timestamps.")
    parser.add_argument('--prompt_videoqa', type=str, default="Question: What does this TV news report about?\nOptions:\n(A) thievery\n(B) community violence incidents\n(C) fashion show\n(D) aging population")
    parser.add_argument('--prompt_referring', type=str, default="What is happening from 70 seconds to 80 seconds?")
    parser.add_argument('--video_path', type=str, default="./experiments/_3klvlS4W7A.mp4")

    # generation
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=None)
    args = parser.parse_args()
    return args

def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def create_inputs(args, mode):
    """
    video_input
    """
    video_processor = frame_transform(image_size=224, mean=INTERNVIDEO_MEAN, std=INTERNVIDEO_STD)
    image_processor = frame_transform(image_size=336, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
    pixel_values, frame_indices, fps, total_frame_num, duration = read_frames_decord(
        video_path = args.video_path,
        num_frames = args.num_frames,
        sample = 'middle',
    )
    temporal_pixel_values = []
    for i in range(pixel_values.shape[0]): 
        temporal_pixel_values.append(video_processor(pixel_values[i]))
    temporal_pixel_values = torch.tensor(np.array(temporal_pixel_values)) # [num_frames, 3, 224, 224]
    temporal_pixel_values = temporal_pixel_values.unsqueeze(0)

    num_frames_per_seg = int(args.num_frames // args.num_segs)
    indices_spatial = [(i*num_frames_per_seg) + int(num_frames_per_seg/2) for i in range(args.num_segs)]
    spatial_pixel_values = []
    for i_spatial in indices_spatial:
        spatial_pixel_values.append(image_processor(pixel_values[i_spatial]))
    spatial_pixel_values = torch.tensor(np.array(spatial_pixel_values)) # [num_segs, 3, 336, 336]
    spatial_pixel_values = spatial_pixel_values.unsqueeze(0)

    """
    text_input
    """
    assert mode in ['qa', 'grounding', 'referring']
    chat_template = {'phi3.5': Phi_3_5_Template(), 'llama3': LLaMA3_Template(), 'vicuna': Vicuna_Template()}[args.llm]
    if mode == 'grounding':
        conv = [
            {"from": "human", "value": DEFAULT_IMAGE_TOKEN + ' ' + GROUNDING_TOKEN + '\n'+args.prompt_grounding},
            {"from": "gpt", "value": ''}                
        ]
    elif mode == 'qa':
        conv = [
            {"from": "human", "value": DEFAULT_IMAGE_TOKEN + '\n'+ args.prompt_videoqa},
            {"from": "gpt", "value": ''}                
        ]
    elif mode == 'referring':
        query = args.prompt_referring
        query = re.sub(r'(\d+) seconds', lambda m: f"<{int(float(m.group(1))/duration*args.num_temporal_tokens)}>", query)
        conv = [
            {"from": "human", "value": DEFAULT_IMAGE_TOKEN + '\n'+ query},
            {"from": "gpt", "value": ''}                
        ]
    sep, eos = chat_template.separator.apply()
    prompt = chat_template.encode(conv).replace(eos, '')

    samples = {
            "video_ids": [args.video_path],
            "question_ids": [args.video_path],
            "prompts": [prompt],
            "temporal_pixel_values": temporal_pixel_values.to(args.device),
            "spatial_pixel_values": spatial_pixel_values.to(args.device),
        }
    
    return samples, duration

def parse_time_interval(text, duration, num_temporal_tokens=300, llm='phi3.5'):
    pattern = r"<(\d+)>"
    def replace_func(match):
        x = int(match.group(1))
        m = duration * x / num_temporal_tokens
        if llm=='phi3.5':
            return f" {m:.2f} seconds"
        elif llm=='llama3':
            return f"{m:.2f} seconds"
    return re.sub(pattern, replace_func, text)


if __name__ == '__main__':
    args = parse_args()
    init_seeds(args.seed)

    model = LLAVA_NEXT_VIDEO(
        dtype=args.dtype, 
        stage=args.stage, 
        max_txt_len=args.max_txt_len, 
        num_frames=args.num_frames,
        num_segs=args.num_segs,
        num_temporal_tokens=args.num_temporal_tokens,
        lora=args.lora,
        llm=args.llm,
        attn_implementation=args.attn_implementation,
        config_path=args.config_path,
        tokenizer_path=args.tokenizer_path,
        pretrained_video_path=args.pretrained_video_path,
        pretrained_vision_proj_llm_path=args.pretrained_vision_proj_llm_path, 
        )
    ckpt = torch.load(args.ckpt_path, map_location='cpu')['model']
    if 'multi_modal_projector' in ckpt.keys():
        model.multi_modal_projector.load_state_dict(ckpt['multi_modal_projector'])
    if 'video_projecter' in ckpt.keys():
        model.video_projecter.load_state_dict(ckpt['video_projecter'])
    if 'language_model' in ckpt.keys():
        model.language_model.load_state_dict(ckpt['language_model'])  
    model.eval()
    model.to(args.device)

    samples_grounding, duration_grounding = create_inputs(args, 'grounding')
    samples_videoqa, duration_videoqa = create_inputs(args, 'qa')
    samples_referring, duration_referring = create_inputs(args, 'referring')

    generate_kwargs = {
    "do_sample": args.do_sample,
    "num_beams": args.num_beams,
    "max_new_tokens": args.max_new_tokens,
    "temperature":args.temperature,
    "top_p":args.top_p,
    }

    with torch.cuda.amp.autocast(enabled=True, dtype=model.dtype):
        with torch.inference_mode():
            pred_texts_grounding = model.generate(samples_grounding, **generate_kwargs)[0]
            pred_texts_videoqa = model.generate(samples_videoqa, **generate_kwargs)[0]
            pred_texts_referring = model.generate(samples_referring, **generate_kwargs)[0]

    print('\n******grounding example******')
    print(samples_grounding['prompts'][0])
    print(parse_time_interval(pred_texts_grounding, duration_grounding, args.num_temporal_tokens, args.llm))

    print('\n******referring example******')
    print(samples_referring['prompts'][0])
    print(pred_texts_referring)

    print('\n******videoqa example******')
    print(samples_videoqa['prompts'][0])
    print(pred_texts_videoqa)