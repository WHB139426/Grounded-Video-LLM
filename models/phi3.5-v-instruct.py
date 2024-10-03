import sys
import os
import contextlib
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import copy
from transformers import AutoTokenizer, AutoConfig, CLIPVisionConfig
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from datasets.chat.base_template import IMAGE_TOKEN_INDEX, IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, LLaMA3_Template, Vicuna_Template, Phi_3_5_Template
from models.modeling_phi3 import Phi3ForCausalLM, Phi3Config
from models.modeling_clip import CLIPVisionModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Phi3_5_Projecter(nn.Module):
    def __init__(self, hidden_size=1024*4, intermediate_size=3072):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.linear_0 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_1 = nn.Linear(self.intermediate_size, self.intermediate_size, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, x):
        x = self.linear_0(x)
        x = self.act_fn(x)
        x = self.linear_1(x)
        return x

CLIP_VIT_LARGE_PATCH14_336_CONFIG = CLIPVisionConfig(
  attention_dropout=0.0,
  dropout=0.0,
  hidden_act="quick_gelu",
  hidden_size=1024,
  image_size=336,
  initializer_factor=1.0,
  initializer_range=0.02,
  intermediate_size=4096,
  layer_norm_eps=1e-05,
  num_attention_heads=16,
  num_channels=3,
  num_hidden_layers=24,
  patch_size=14,
  projection_dim=768
)

class Phi3_5_V(nn.Module):
    def __init__(
        self,
        dtype=torch.bfloat16,
        stage='pretrain',
        max_txt_len=2048,
        lora=False,
        llm='phi3.5',
    ):
        super().__init__()
        self.dtype = dtype
        self.max_txt_len = max_txt_len
        self.stage = stage
        self.lora = lora
        self.llm = llm

        if self.llm == 'phi3.5':
            self.config = AutoConfig.from_pretrained("/home/haibo/weights/Phi-3.5-vision-instruct", trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained("/home/haibo/weights/Phi-3.5-mini-instruct")
            self.tokenizer.pad_token = '<|end|>' # 32007
            self.separator = Phi_3_5_Template.separator

        print("loading vision_tower")
        vision_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
        vision_config.torch_dtype = self.dtype
        self.vision_tower = CLIPVisionModel(vision_config)
        if self.llm == 'phi3.5':
            self.vision_tower.load_state_dict(torch.load('/home/haibo/weights/Phi-3.5-vision-instruct-seperated/vision_model.pth', map_location='cpu'))
            image_newlines = torch.load('/home/haibo/weights/Phi-3.5-vision-instruct-seperated/image_newlines.pth', map_location='cpu')
            self.glb_GN = image_newlines['glb_GN'].to(self.device)
            self.sub_GN = image_newlines['sub_GN'].to(self.device)

        print("loading multi_modal_projector")
        if self.llm == 'phi3.5':
            self.multi_modal_projector = Phi3_5_Projecter()
            self.multi_modal_projector.load_state_dict(torch.load('/home/haibo/weights/Phi-3.5-vision-instruct-seperated/multi_modal_projector.pth', map_location='cpu'))

        print("loading language_model")
        if self.llm == 'phi3.5':
            self.language_model = Phi3ForCausalLM.from_pretrained('/home/haibo/weights/Phi-3.5-vision-instruct-seperated/language_model_seperated', torch_dtype=self.dtype, use_cache=False)

        self.all_module_keys = ["vision_tower", "language_model", "multi_modal_projector"]

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def maybe_autocast(self):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=self.dtype)
        else:
            return contextlib.nullcontext()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        sep, eos_token = self.separator.apply()
        total_len = int(labels.ne(tokenizer.pad_token_id).sum())
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            total_len += prompt.count(eos_token)
        rounds = prompt.split(eos_token)
        eos_token_length = 1
        if self.llm == 'phi3.5':
            labels, cur_len = self._make_masks_phi3(labels, tokenizer, sep, eos_token_length, rounds)
        if cur_len != total_len:
            print(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
            )
        return labels

    def _make_masks_phi3(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 1 # bos
        eos_token_length = 1
        bos_token_length = 1
        labels[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length - bos_token_length
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1 - bos_token_length
            if i >=1:
                instruction_len += 1
                round_len += 1
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len

    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        def _insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in _insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).to(self.device)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def prepare_batch(self, text_inputs):
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        for text in text_inputs:
            input_ids = self.tokenizer_image_token(text, self.tokenizer, return_tensors='pt')
            labels = self.make_labels(input_ids, text, self.tokenizer).to(self.device)
            attention_mask = torch.ones(input_ids.shape[0], dtype=torch.long).to(self.device)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)

        # Pad the sequences
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=IGNORE_INDEX).to(self.device)
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(batch_attention_mask, batch_first=True, padding_value=0).to(self.device)

        # Truncate the sequences
        if batch_input_ids.shape[1] > self.max_txt_len:
            batch_input_ids = batch_input_ids[:, :self.max_txt_len]
            batch_labels = batch_labels[:, :self.max_txt_len]
            batch_labels[:, -1] = self.tokenizer.eos_token_id
            batch_attention_mask = batch_attention_mask[:, :self.max_txt_len]

        # print(batch_input_ids)
        # for labels in batch_labels:
        #     print(labels)
        #     test_labels = []
        #     for i in labels:
        #         if i != -100:
        #             test_labels.append(self.tokenizer.decode(torch.tensor([i]), skip_special_tokens=False))
        #         else:
        #             test_labels.append(-100)
        #     print(test_labels)

        return batch_input_ids, batch_labels, batch_attention_mask

    def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop):
        """
        image_features: (num_images*num_crops, 24*24, 1024)
        output: (num_images, h_crop*12, w_crop*12, 4096), h_crop*w_crop == num_crops
        """
        N, L, C = image_features.shape
        assert L == 24 * 24 and C == 1024 and N % (h_crop * w_crop) == 0
        num_images = N // (h_crop * w_crop)
        H = int(L**0.5)
        image_features_hd = (
            image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
            .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
            .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
            .reshape(N, -1, 4 * C)  # N, 144, 4096
            .reshape(
                num_images, h_crop, w_crop, H // 2, H // 2, -1
            )  # n_img, h_crop, w_crop, 12, 12, 4096
            .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
            .reshape(
                num_images, h_crop * H // 2, w_crop * H // 2, 4 * C
            )  # n_img, h_crop*12, w_crop*12, 4096
        )
        return image_features_hd

    def add_image_newline(self, image_features_hd):
        """
        image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
        output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
        """
        num_images, h, w, hid_dim = image_features_hd.shape
        # add the newline token to the HD image feature patches
        newline_embeddings = self.sub_GN.expand(num_images, h, -1, -1).to(self.device)  # (n_img, h, 1, hid_dim)
        image_features_hd_newline = torch.cat(
            [image_features_hd, newline_embeddings], dim=2
        ).reshape(num_images, -1, hid_dim)
        return image_features_hd_newline
    
    def encode_images(self, pixel_values):
        vision_feature_layer = -2
        vision_feature_select_strategy = "default"
        with self.maybe_autocast():
            image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:] # [bs, 576, 1024]
            elif vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature 

            selected_image_feature = self.reshape_hd_patches_2x2merge(selected_image_feature, 1, 1) # [bs, 12, 12, 4096]
            selected_image_feature = self.add_image_newline(selected_image_feature) # [bs, 156, 4096]
            # bs, h, w, d = selected_image_feature.shape
            # selected_image_feature = selected_image_feature.reshape(bs, -1, d) # [bs, 144, 4096]
            image_features = self.multi_modal_projector(selected_image_feature) # [bs, 144, 3072]

        return image_features

    def prepare_multimodal_inputs(self, batch_input_ids, batch_labels, batch_attention_mask, batch_image_features, batch_image_ids):
        new_input_embeds = []
        new_labels = []
        new_attention_masks = []
        for image_embeds, input_ids, labels, attention_mask, image_ids in zip(batch_image_features, batch_input_ids, batch_labels, batch_attention_mask, batch_image_ids):
            """
            image_embeds: [576, 3072]
            input_ids: [seq]
            labels: [seq]
            attention_mask: [seq]
            """
            image_index = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]
            pre_embeds = self.get_input_embeddings()(input_ids[:image_index].unsqueeze(0))[0]
            post_embeds = self.get_input_embeddings()(input_ids[image_index+1:].unsqueeze(0))[0]

            if image_ids == 'text':
                new_input_embeds.append(torch.cat([pre_embeds, post_embeds, image_embeds], dim=0))
                new_labels.append(torch.cat([labels[:image_index], labels[image_index+1:], torch.ones(image_embeds.shape[0], dtype=torch.long).to(self.device)*IGNORE_INDEX], dim=0))
                new_attention_masks.append(torch.cat([attention_mask[:image_index], attention_mask[image_index+1:], torch.zeros(image_embeds.shape[0], dtype=torch.long).to(self.device)], dim=0))
            else:
                new_input_embeds.append(torch.cat([pre_embeds, image_embeds, post_embeds], dim=0))
                new_labels.append(torch.cat([labels[:image_index], torch.ones(image_embeds.shape[0], dtype=torch.long).to(self.device)*IGNORE_INDEX, labels[image_index+1:]], dim=0))
                new_attention_masks.append(torch.cat([attention_mask[:image_index], torch.ones(image_embeds.shape[0], dtype=torch.long).to(self.device), attention_mask[image_index+1:]], dim=0))

        new_input_embeds = torch.stack(new_input_embeds, dim=0).to(self.device)
        new_labels = torch.stack(new_labels, dim=0).to(self.device)
        new_attention_masks = torch.stack(new_attention_masks, dim=0).to(self.device)

        return new_input_embeds, new_labels, new_attention_masks

    def forward(
        self,
        samples,
    ):
        with self.maybe_autocast():
            batch_input_ids, batch_labels, batch_attention_mask = self.prepare_batch(samples['text_inputs'])
            batch_image_features = self.encode_images(samples['pixel_values'])
            inputs_embeds, labels, attention_masks = self.prepare_multimodal_inputs(batch_input_ids, batch_labels, batch_attention_mask, batch_image_features, samples['image_ids'])

            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_masks,
                return_dict=True,
                labels=labels,
            )
        loss = outputs.loss
        return {"loss": loss}

    @torch.inference_mode()
    def generate(
        self,
        samples,
        **generate_kwargs
    ):
        prompts = samples['prompts']
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        for text in prompts:
            input_ids = self.tokenizer_image_token(text, self.tokenizer, return_tensors='pt')
            labels = copy.deepcopy(input_ids)
            attention_mask = torch.ones(input_ids.shape[0], dtype=torch.long).to(self.device)
            batch_input_ids.append(torch.flip(input_ids, dims=[0])) # reverse the sequence
            batch_labels.append(labels)
            batch_attention_mask.append(torch.flip(attention_mask, dims=[0])) # reverse the sequence
        
        # Pad the sequences
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=IGNORE_INDEX).to(self.device)
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(batch_attention_mask, batch_first=True, padding_value=0).to(self.device)

        # Truncate the sequences
        if batch_input_ids.shape[1] > self.max_txt_len:
            batch_input_ids = batch_input_ids[:, :self.max_txt_len]
            batch_labels = batch_labels[:, :self.max_txt_len]
            batch_attention_mask = batch_attention_mask[:, :self.max_txt_len]

        # Reverse the sequence back
        batch_input_ids = torch.flip(batch_input_ids, dims=[1])
        batch_attention_mask = torch.flip(batch_attention_mask, dims=[1])

        # image_features
        batch_image_features = self.encode_images(samples['pixel_values'])

        inputs_embeds, labels, attention_masks = self.prepare_multimodal_inputs(batch_input_ids, batch_labels, batch_attention_mask, batch_image_features, samples['image_ids'])

        with self.maybe_autocast():
            outputs = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                eos_token_id=self.tokenizer.eos_token_id,
                attention_mask=attention_masks,
                pad_token_id=self.tokenizer.pad_token_id,
                **generate_kwargs,
            )

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text




# device = "cuda:0"
# # device = "cpu"
# dtype = torch.float32 if device == 'cpu' else torch.bfloat16
# llm = 'phi3.5'

# image_processor = image_transform(image_size=336)
# pixel_values = torch.stack([image_processor(load_image('/home/haibo/workspace/000000039769.jpg')), image_processor(load_image('/home/haibo/workspace/australia.jpg'))], dim=0)
# chat_template = Phi_3_5_Template()


# prompt_conversations = [
#     {"from": "human", "value": "<image>\n"+'Describe the image.'},
#     {"from": "gpt", "value": ''}
# ]
# sep, eos = chat_template.separator.apply()
# prompt = chat_template.encode(prompt_conversations).replace(eos, '')

# text_input_conversations = [
#     {"from": "human", "value": "<image>\n"+'Provide a one-sentence caption for the provided image.'},
#     {"from": "gpt", "value": 'This is an image.'},
#     {"from": "human", "value": "Can you show me more details?"},
#     {"from": "gpt", "value": "No problem."},
#     {"from": "human", "value": "What can i say."},
#     {"from": "gpt", "value": "Mamba out."},
# ]
# text_input = chat_template.encode(text_input_conversations)

# prompt_conversations_copy = [
#     {"from": "human", "value": "<image>\n"+'What is so special about this image?'},
#     {"from": "gpt", "value": ''}
# ]
# sep, eos = chat_template.separator.apply()
# prompt_copy = chat_template.encode(prompt_conversations_copy).replace(eos, '')


# text_input_conversations_copy = [
#     {"from": "human", "value": "<image>\n"+'Show me the truth'},
#     {"from": "gpt", "value": 'I cannot.'},
#     {"from": "human", "value": "Can you show me more details?"},
#     {"from": "gpt", "value": "This is a sad story."},
# ]
# text_input_copy = chat_template.encode(text_input_conversations_copy)




# model = Phi3_5_V(dtype=dtype, llm=llm)
# model.to(device)
# print(get_parameter_number(model))

# samples = {
#         "image_ids": ['xxxx', 'xxxx'],
#         "text_inputs": [text_input, text_input_copy],
#         "prompts": [prompt, prompt_copy],
#         "pixel_values": pixel_values.to(device),
#     }

# loss = model(samples)
# print(loss)

# with torch.inference_mode():
#     generate_kwargs = {
#         "do_sample": False,
#         "num_beams": 1, 
#         "max_new_tokens": 256,
#         "temperature":1,
#         }
#     output_text = model.generate(samples, **generate_kwargs)

# print(samples['text_inputs'])
# print(samples['prompts'])
# print(output_text)



























# device = 'cuda:0'

# chat_template = Phi_3_5_Template()
# conversations = [
#     {"from": "human", "value": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
#     {"from": "gpt", "value": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
#     {"from": "human", "value": "What about solving an 2x + 3 = 7 equation?"},
#     {"from": "gpt", "value": ""}
# ]
# sep, eos = chat_template.separator.apply()
# prompts = chat_template.encode(conversations).replace(eos, '')
# print(prompts)

# tokenizer = AutoTokenizer.from_pretrained("/home/haibo/weights/Phi-3.5-mini-instruct")
# model = Phi3ForCausalLM.from_pretrained("/home/haibo/weights/Phi-3.5-mini-instruct").to(device)

# inputs = tokenizer(
#             [prompts],
#             return_tensors="pt",
#             padding="longest",
#             truncation=True,
#             max_length=1024,
#         ).to(device)
# input_ids = inputs.input_ids

# outputs = model.generate(input_ids, max_new_tokens=2048)
# output_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
# print(output_text)
