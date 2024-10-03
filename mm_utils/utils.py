import requests
from PIL import Image
from io import BytesIO
import json
import pandas as pd
import pickle
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, CenterCrop, ToPILImage
from typing import Optional, Tuple, Any, Union, List


dense_caption_prompts_detail = [
    "Detect and list each event in detail and its corresponding timestamps that appears in the video.",
    "Determine the start and end times of all activity events in detail, accompanied by descriptions.",
    "Capture and describe the activity events in detail, specifying their respective time intervals.",
    "Identify, timestamp, and describe various activity events occurring in the video without omission. The timestamp should include the start time and end time in seconds.",
    "Examine the video and enumerate all events you can see in detail, together with their start and end times.",
    "Perform a thorough analysis of the video and list out every event in detail with its timestamps.",
    "In the provided video, pinpoint and list all the events in detail, together with their respective time intervals.",
    "Could you outline all the events in detail and their timestamps that are visible within the video?",
]

dense_caption_prompts_short = [
    "Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences.",
    "Detect and report the start and end timestamps of activity events in the video, along with descriptions.",
    "Pinpoint the time intervals of activity events in the video, and provide descriptions for each event.",
    "Can you compile a list of the activities and their timestamps featured in the video?",
    "I need you to scrutinize the video and catalog every event it contains, along with the timestamps.",
]

dense_caption_prompts_one_end_short = [
    "Localize a series of activity events in the video, output the one single timestamp for each event, and describe each event with sentences.",
    "Detect and report the point of time of activity events in the video, along with descriptions.",
    "Pinpoint the point of time of activity events in the video, and provide descriptions for each event.",
    "Can you compile a list of the activities and their point of time featured in the video?",
    "I need you to scrutinize the video and catalog every event it contains, along with one single timestamp.",
    "I need you to scrutinize the video and catalog every event it contains, along with the point of time.",
]

sft_step_prompts_one_end = [
    "Localize a series of action steps in the given video, output the one single timestamp for each step, and briefly describe the step.",
    "Locate and describe a series of actions or steps in the video, including their point of time.",
    "Identify and mark the video segments corresponding to a series of actions or steps, specifying the point of time and describing the steps.",
    "Find, identify, and determine the temporal boundaries of a series of distinct actions or steps occurring throughout the video. For each action, output the corresponding one single timestamp, accompanied by a concise description.",
    "Identify and localize a series of steps or actions occurring in the video, providing one single timestamp and related descriptions.",
    "Locate and pinpoint a sequential series of specific actions or steps in the video, accurately specifying the point of time for each action. Additionally, provide a succinct description of each action."
]

short_caption_prompts = [
    "Describe the following video concisely.", 
    "Provide a brief description of the given video clip.", 
    "Offer a succinct explanation of the footage presented.", 
    "Summarize the visual content of the following video.", 
    "Give a short and clear explanation of the subsequent video clip.", 
    "Share a concise interpretation of the video provided.", 
    "Present a compact description of the clip's key features.", 
    "Relay a brief, clear account of the video shown.", 
    "Render a clear and concise summary of the video below.", 
    "Write a terse but informative summary of the following video clip.", 
    "Create a compact narrative representing the video presented.",
]

detail_caption_prompts = [
    "Summarize the video content thoroughly.",
    "What does this video depict in detail?",
    "How would you detail the actions shown in the video?",
    "Create an in-depth description of the events occurring in the video.",
    "Provide a detailed description of the events taking place in this video.",
    "Offer a detailed analysis of this video.",
    "Can you generate a comprehensive caption for this video?",
    "Can you give me a detailed description of the provided video?",
    "Describe this video in a detailed manner.",
]

vtg_prompts = [
    "When does '%s' happen in the video?",
    "At what time does the occurrence of '%s' take place in the video?",
    "During which part of the video does '%s' occur?",
    "When in the video does the '%s' incident occur?",
    "At which moment does '%s' take place in the video?",
    "During which phase of the video does '%s' happen?",
    "When does the '%s' event occur in the video?",
    "At what time does '%s' occur in the video sequence?",
    "When does the '%s' situation take place in the video?",
    "At which time interval in the video can we see '%s' occurring?",
]

vtu_prompts = [
    "What is happening from <start> to <end>?",
    "What is taking place between <start> and <end>?",
    "What events unfold between <start> and <end>?",
    "What is happening during the period from <start> to <end>?",
    "What occurs between <start> and <end>?",
    "What is going on from <start> to <end>?",
    "How do things progress from <start> to <end>?",
    "Can you describe what happens from <start> to <end>?",
    "Describe the events occurring between <start> and <end>.",
    "Narrate the actions that unfold from <start> to <end>.",
    "Summarize the happenings between <start> and <end>.",
    "Identify the main activities occurring from <start> to <end>.",
    "Provide an overview of what happens from <start> to <end>.",
]

sft_vtg_two_end_prompts = [
    "Localize the visual content described by the given textual query <query_placeholder> in the video, and output the start and end timestamps.",
    "Detect and report the start and end timestamps of the video segment that semantically matches the given textual query <query_placeholder>.",
    "Give you a textual query: <query_placeholder>. When does the described content occur in the video? Please return the start and end timestamps.",
    "Locate the visual content mentioned in the text query <query_placeholder> within the video, including start and end timestamps.",
    "The given natural language query <query_placeholder> is semantically aligned with a video moment, please give the start time and end time of the video moment.",
    "Find the video segment that corresponds to the given textual query <query_placeholder> and determine its start and end positions."
]

sft_vtg_one_end_prompts = [
    "Localize the visual content described by the given textual query <query_placeholder> in the video, and output one single timestamp.",
    "Detect and report the point of time of the video segment that semantically matches the given textual query <query_placeholder>.",
    "Give you a textual query: <query_placeholder>. When does the described content occur in the video? Please return the point of time.",
    "Locate the visual content mentioned in the text query <query_placeholder> within the video, including the point of time.",
    "The given natural language query <query_placeholder> is semantically aligned with a video moment, please give the single timestamp point of the video moment.",
    "Find the video segment that corresponds to the given textual query <query_placeholder> and determine its single timestamp point."
]

sft_specific_step_prompts = [
    "Localize a series of action steps from <start> to <end>, output a start and end timestamp for each step, and briefly describe the step.",
    "Locate and describe a series of actions or steps in the video from <start> to <end>, including their start and end timestamps.",
    "Identify and mark the video segments corresponding to a series of actions or steps between <start> and <end>, specifying the timestamps and describing the steps.",
    "Find, identify, and determine the temporal boundaries of a series of distinct actions or steps occurring between <start> and <end>. For each action, output the corresponding start and end timestamps, accompanied by a concise description.",
    "Identify and localize a series of steps or actions occurring in the video from <start> to <end>, providing start and end timestamps and related descriptions.",
    "Locate and pinpoint a sequential series of specific actions or steps in the video between <start> and <end>, accurately specifying the start and end timestamps for each action. Additionally, provide a succinct description of each action."
]

sft_step_prompts = [
    "Localize a series of action steps in the given video, output a start and end timestamp for each step, and briefly describe the step.",
    "Locate and describe a series of actions or steps in the video, including their start and end timestamps.",
    "Identify and mark the video segments corresponding to a series of actions or steps, specifying the timestamps and describing the steps.",
    "Find, identify, and determine the temporal boundaries of a series of distinct actions or steps occurring throughout the video. For each action, output the corresponding start and end timestamps, accompanied by a concise description.",
    "Identify and localize a series of steps or actions occurring in the video, providing start and end timestamps and related descriptions.",
    "Locate and pinpoint a sequential series of specific actions or steps in the video, accurately specifying the start and end timestamps for each action. Additionally, provide a succinct description of each action."
]



def _convert_to_rgb(image):
    return image.convert('RGB')

SIGLIP_DATASET_MEAN = (0.5, 0.5, 0.5)
SIGLIP_DATASET_STD = (0.5, 0.5, 0.5)

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

INTERNVIDEO_MEAN = (0.485, 0.456, 0.406)
INTERNVIDEO_STD = (0.229, 0.224, 0.225)

def frame_transform(
        image_size: int,
        rescale_factor: float = 1.0,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3
    
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    
    transforms = [
        ToPILImage(),
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
    ]
    transforms.extend([
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])
    return Compose(transforms)

import torch
class Rescale:
    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.factor
    
def image_transform(
        image_size: int,
        rescale_factor: float = 1.0,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3
    
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    
    transforms = [
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
    ]

    transforms.extend([
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])
    return Compose(transforms)

def expand2square(pil_img, background_color=tuple(int(x*255) for x in OPENAI_DATASET_MEAN)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def load_image(image_file, pad=False):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    if pad:
        image = expand2square(image)
    return image

def load_txt(path):
    strings_list = []
    with open(path, 'r') as file:
        for line in file:
            # 去除每行的换行符，并将其添加到列表中
            strings_list.append(line.strip())
    return strings_list

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)
        
def load_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)
    return data

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_csv(path):
    file_list = []
    data = pd.read_csv(path)
    columns = data.columns.tolist()
    for index, row in data.iterrows():
        file_list.append({})
        for column in columns:
            file_list[index][column] = row[column]
    return file_list

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num} 


