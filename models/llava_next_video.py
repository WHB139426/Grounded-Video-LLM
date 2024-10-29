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
import einops
import math
from einops import rearrange
from typing import Callable
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from datasets.chat.base_template import IMAGE_TOKEN_INDEX, IGNORE_INDEX, GROUNDING_TOKEN, DEFAULT_IMAGE_TOKEN, LLaMA3_Template, Vicuna_Template, Phi_3_5_Template
from models.modeling_llama import LlamaForCausalLM
from models.modeling_phi3 import Phi3ForCausalLM
from models.modeling_clip import CLIPVisionModel
from models.internvideo2 import pretrain_internvideo2_1b_patch14_224, interpolate_pos_embed_internvideo2_new
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Video_Projecter(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.intermediate_size, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, x):
        x = self.up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x

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

class LLAVA_NEXT_VIDEO(nn.Module):
    def __init__(
        self,
        dtype=torch.bfloat16,
        stage='pretrain',
        max_txt_len=2048,
        num_frames=96,
        num_segs=12,
        lora=False,
        num_temporal_tokens=300,
        llm='llama3',
        attn_implementation="flash_attention_2",
        config_path = "weight_path/Phi-3.5-vision-instruct",
        tokenizer_path = "weight_path/Phi-3.5-mini-instruct",
        pretrained_video_path = 'weight_path/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt',
        pretrained_vision_proj_llm_path = 'weight_path/Phi-3.5-vision-instruct-seperated/',        
    ):
        super().__init__()
        self.dtype = dtype
        self.max_txt_len = max_txt_len
        self.num_frames = num_frames
        self.num_segs = num_segs
        self.stage = stage
        self.lora = lora
        self.num_temporal_tokens = num_temporal_tokens
        self.llm = llm

        if self.llm == 'llama3':
            self.config = AutoConfig.from_pretrained(config_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, truncation_side="left")
            self.tokenizer.eos_token_id = 128009 # '<|eot_id|>'
            self.tokenizer.pad_token_id = 128001 # '<|end_of_text|>'
            self.separator = LLaMA3_Template.separator
        elif self.llm == 'vicuna':
            self.config = AutoConfig.from_pretrained(config_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, truncation_side="left")
            self.separator = Vicuna_Template.separator
        elif self.llm == 'phi3.5':
            self.config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
            self.config.vision_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.pad_token = '<|end|>' # 32007
            self.separator = Phi_3_5_Template.separator

        print("loading vision_tower")
        self.config.vision_config.torch_dtype = self.dtype
        self.vision_tower = CLIPVisionModel(self.config.vision_config)
        if self.llm == 'llama3' or self.llm == 'vicuna':
            self.vision_tower.load_state_dict(torch.load(os.path.join(pretrained_vision_proj_llm_path, 'vision_model.pth'), map_location='cpu'))
            self.image_newline = torch.load(os.path.join(pretrained_vision_proj_llm_path, 'image_newline.pth'), map_location='cpu')['image_newline'].to(self.dtype)
        elif self.llm == 'phi3.5':
            self.vision_tower.load_state_dict(torch.load(os.path.join(pretrained_vision_proj_llm_path, 'vision_model.pth'), map_location='cpu'))
            image_newlines = torch.load(os.path.join(pretrained_vision_proj_llm_path, 'image_newlines.pth'), map_location='cpu')
            self.glb_GN = image_newlines['glb_GN'].to(self.device)
            self.sub_GN = image_newlines['sub_GN'].to(self.device)

        print("loading video_encoder")
        self.video_encoder = pretrain_internvideo2_1b_patch14_224(num_frames=self.num_frames//self.num_segs, use_flash_attn=True if attn_implementation=="flash_attention_2" else False)
        state_dict = torch.load(pretrained_video_path, map_location='cpu')
        interpolate_pos_embed_internvideo2_new(state_dict, self.video_encoder, orig_t_size=4)
        self.video_encoder.load_state_dict(state_dict, strict=True)
        self.video_encoder.to(self.dtype)

        print("loading multi_modal_projector")
        if self.llm == 'llama3' or self.llm == 'vicuna':
            self.multi_modal_projector = LlavaMultiModalProjector(self.config)
            self.multi_modal_projector.load_state_dict(torch.load(os.path.join(pretrained_vision_proj_llm_path, 'multi_modal_projector.pth'), map_location='cpu'))
        elif self.llm == 'phi3.5':
            self.multi_modal_projector = Phi3_5_Projecter()
            self.multi_modal_projector.load_state_dict(torch.load(os.path.join(pretrained_vision_proj_llm_path, 'multi_modal_projector.pth'), map_location='cpu'))

        print("loading video_projector")
        self.video_projecter = Video_Projecter(1408, self.config.hidden_size)

        print("loading language_model")
        if self.llm == 'llama3' or self.llm == 'vicuna':
            self.language_model = LlamaForCausalLM.from_pretrained(os.path.join(pretrained_vision_proj_llm_path, 'language_model_seperated'), torch_dtype=self.dtype, use_cache=False, attn_implementation=attn_implementation)
        elif self.llm == 'phi3.5':
            self.language_model = Phi3ForCausalLM.from_pretrained(os.path.join(pretrained_vision_proj_llm_path, 'language_model_seperated'), torch_dtype=self.dtype, use_cache=False, attn_implementation=attn_implementation)

        self.all_module_keys = ["vision_tower", "language_model", "video_encoder", "multi_modal_projector", "video_projecter"]

        if self.stage == 'pretrain':
            print("Frozen vision_tower")
            for name, param in self.vision_tower.named_parameters():
                param.requires_grad = False
            print("Frozen video_encoder")
            for name, param in self.video_encoder.named_parameters():
                param.requires_grad = False
            print("Frozen LLM")
            for name, param in self.language_model.named_parameters():
                param.requires_grad = False
            self.trainable_module_keys = ["multi_modal_projector", "video_projecter"]

        elif self.stage == 'grounded':
            print("Frozen ViT")
            for name, param in self.vision_tower.named_parameters():
                param.requires_grad = False
            print("Frozen video_encoder")
            for name, param in self.video_encoder.named_parameters():
                param.requires_grad = False

            self.reset_embeddings()

            if self.lora:
                print("LORA llm")
                self.lora_model()

            print("Frozen Part LLM")
            for name, param in self.language_model.named_parameters():
                if 'lm_head' in name or 'embed_tokens' in name or 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.trainable_module_keys = ["multi_modal_projector", "video_projecter", "language_model"]

        elif self.stage == 'sft':
            print("Frozen ViT")
            for name, param in self.vision_tower.named_parameters():
                param.requires_grad = False
            print("Frozen video_encoder")
            for name, param in self.video_encoder.named_parameters():
                param.requires_grad = False

            self.reset_embeddings()

            if self.lora:
                print("LORA llm")
                self.lora_model()

            print("Frozen Part LLM")
            for name, param in self.language_model.named_parameters():
                if 'lm_head' in name or 'embed_tokens' in name or 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            self.trainable_module_keys = ["multi_modal_projector", "video_projecter", "language_model"]  
        
    def lora_model(self,):
        from peft import get_peft_model, LoraConfig, TaskType
        if self.llm == 'phi3.5':
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False if self.training else True, r=128, lora_alpha=256, lora_dropout=0.05, 
                target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
            )
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False if self.training else True, r=128, lora_alpha=256, lora_dropout=0.05, 
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", 'gate_proj'],
            )
        self.language_model = get_peft_model(self.language_model, peft_config)

        for name, module in self.language_model.named_modules(): 
            if "lora" in name.lower(): 
                for param in module.parameters(): 
                    param.data = param.data.to(self.dtype)

    def reset_embeddings(self,):
        """
        tokenizer and embed
        """
        special_token_list = [f'<{i}>' for i in range(self.num_temporal_tokens + 1)] + [GROUNDING_TOKEN]
        self.tokenizer.add_tokens(special_token_list)
        num_new_tokens = len(special_token_list)
        self.language_model.config.vocab_size += num_new_tokens

        """
        word embeddings
        """
        embedding_layer = self.language_model.get_input_embeddings()
        average_embedding = torch.mean(embedding_layer.weight, dim=0)

        old_num_tokens, old_embedding_dim = embedding_layer.weight.shape
        
        new_embeddings = nn.Embedding(old_num_tokens + num_new_tokens, old_embedding_dim)
        new_embeddings.to(embedding_layer.weight.device, dtype=embedding_layer.weight.dtype)
        new_embeddings.weight.data[:old_num_tokens, :] = embedding_layer.weight.data[:old_num_tokens, :]
        new_embeddings.weight.data[old_num_tokens:, :] = average_embedding

        self.language_model.set_input_embeddings(new_embeddings)

        """
        lm_head
        """
        lm_head = self.language_model.get_output_embeddings()
        average_head = torch.mean(lm_head.weight, dim=0)

        old_num_tokens, old_hidden_size = lm_head.weight.shape

        new_lm_head = nn.Linear(old_hidden_size, old_num_tokens + num_new_tokens)
        new_lm_head.to(lm_head.weight.device, dtype=lm_head.weight.dtype)
        new_lm_head.weight.data[:old_num_tokens, :] = lm_head.weight.data[:old_num_tokens, :]
        new_lm_head.weight.data[old_num_tokens:, :] = average_head

        self.language_model.set_output_embeddings(new_lm_head)

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_tower.get_fsdp_wrapping_policy()
        video_fsdp_wrapping_policy = self.video_encoder.get_fsdp_wrapping_policy()
        if self.stage == 'grounded':
            llm_fsdp_wrapping_policy = self.language_model.get_fsdp_wrapping_policy_embedding()
        elif self.stage=='pretrain':
            llm_fsdp_wrapping_policy = self.language_model.get_fsdp_wrapping_policy()
        elif self.stage == 'sft':
            llm_fsdp_wrapping_policy = self.language_model.get_fsdp_wrapping_policy_embedding()
        else:
            llm_fsdp_wrapping_policy = self.language_model.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        if self.llm == 'phi3.5':
            prismatic_fsdp_wrapping_policy = partial(
                _module_wrap_policy,
                module_classes={Phi3_5_Projecter, Video_Projecter}, # Video_Projecter
            )
        else:
            prismatic_fsdp_wrapping_policy = partial(
                _module_wrap_policy,
                module_classes={LlavaMultiModalProjector, Video_Projecter}, # Video_Projecter
            )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                video_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

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
        if self.llm == 'llama3':
            labels, cur_len = self._make_masks_llama3(labels, tokenizer, sep, eos_token_length, rounds)
        elif self.llm == 'vicuna':
            labels, cur_len = self._make_masks_vicuna(labels, tokenizer, sep, eos_token_length, rounds)
        elif self.llm == 'phi3.5':
            labels, cur_len = self._make_masks_phi3(labels, tokenizer, sep, eos_token_length, rounds)

        if cur_len != total_len:
            print(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
            )
        return labels
        
    def _make_masks_llama3(self, labels, tokenizer, sep, eos_token_length, rounds):
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
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - bos_token_length
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len

    def _make_masks_vicuna(self, labels, tokenizer, sep, eos_token_length, rounds):
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
                instruction_len -= 1
                round_len -= 1
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len

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
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]
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

        return batch_input_ids, batch_labels, batch_attention_mask

    def reshape_hd_patches_2x2merge_phi3(self, image_features, h_crop, w_crop):
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

    def add_image_newline_phi3(self, image_features_hd):
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
    
    def encode_images(self, samples):

        spatial_pixel_values = samples['spatial_pixel_values'] # [bs, num_segs, 3, 336, 336]
        temporal_pixel_values = samples['temporal_pixel_values'] # [bs, num_frames, 3, 224, 224]

        batch_size, num_segs, _, _, _ = spatial_pixel_values.shape
        batch_size, num_frames, _, _, _ = temporal_pixel_values.shape
        num_frames_per_seg = num_frames//num_segs

        """
        image features
        """
        spatial_pixel_values = rearrange(spatial_pixel_values, "b t c h w -> (b t) c h w") # [bs*num_segs, 3, 336, 336]
        image_outputs = self.vision_tower(spatial_pixel_values, output_hidden_states=True)
        image_features = image_outputs.hidden_states[-2][:, 1:] # [bs*num_segs, 576, 1024]

        if self.llm == 'llama3' or self.llm == 'vicuna':
            # Aadptive Pooling for image features
            def convert_Fembeddings2video(input, num_videos, frame_shape):
                input = einops.rearrange(input, 
                                        '(num_videos num_frames) (h w) embed_dims -> num_videos embed_dims num_frames h w', 
                                        num_videos=num_videos, h=frame_shape[0])
                return input
            frame_shape = (int(math.sqrt(image_features.shape[1])), int(math.sqrt(image_features.shape[1]))) # [24, 24] 
            hidden_states = convert_Fembeddings2video(image_features, batch_size, frame_shape) # [bs, 1024, num_segs, 24, 24] 
            hidden_states = nn.AdaptiveAvgPool3d([num_segs, 8, 8])(hidden_states) # [bs, 1024, num_segs, 8, 8]  
            image_features = einops.rearrange(hidden_states, 'batch_size_num_videos embed_dims num_frames h w -> batch_size_num_videos num_frames (h w) embed_dims', )
            image_features = self.multi_modal_projector(image_features) # [bs, num_segs, 64, 4096] 
        elif self.llm == 'phi3.5':
            image_features = self.reshape_hd_patches_2x2merge_phi3(image_features, 1, 1) # [bs*num_segs, 12, 12, 4096]
            image_features = self.add_image_newline_phi3(image_features) # [bs*num_segs, 156, 4096]
            image_features = einops.rearrange(image_features, '(batch_size num_segs) hw embed_dims -> batch_size num_segs hw embed_dims', batch_size=batch_size, num_segs=num_segs)
            image_features = self.multi_modal_projector(image_features) # [bs, num_segs, 156, 3072]

        """
        segment features
        """
        memroy_efficient = False
        if not memroy_efficient:
            temporal_pixel_values = einops.rearrange(temporal_pixel_values, 'bs (num_segs num_frames_per_seg) c h w -> bs num_segs num_frames_per_seg c h w', num_segs=num_segs)
            temporal_pixel_values = einops.rearrange(temporal_pixel_values, 'bs num_segs num_frames_per_seg c h w -> (bs num_segs) c num_frames_per_seg h w')
            segment_features = self.video_encoder(temporal_pixel_values, None, False, x_vis_return_idx=-2, x_vis_only=True)[:, 1:, :] # [bs*num_segs, num_frames_per_seg*256, 1408]  
            segment_features = einops.rearrange(segment_features, 'bs_num_segs (num_frames_per_seg hw) d -> bs_num_segs num_frames_per_seg hw d', num_frames_per_seg=num_frames_per_seg) # [bs*num_segs, num_frames_per_seg, 256, 1408] 
        else:
            temporal_pixel_values = einops.rearrange(temporal_pixel_values, 'bs (num_segs num_frames_per_seg) c h w -> bs num_segs num_frames_per_seg c h w', num_segs=num_segs)
            temporal_pixel_values = einops.rearrange(temporal_pixel_values, 'bs num_segs num_frames_per_seg c h w -> bs num_segs c num_frames_per_seg h w')
            segment_features = []
            for i in range(temporal_pixel_values.shape[1]):
                segment_features.append(self.video_encoder(temporal_pixel_values[:, i, :, :, :, :], None, False, x_vis_return_idx=-2, x_vis_only=True)[:, 1:, :])
            segment_features = torch.stack(segment_features, dim=1).to(self.device) # [bs, num_segs, num_frames_per_seg*256, 1408] 
            segment_features = einops.rearrange(segment_features, 'bs num_segs (num_frames_per_seg hw) d -> (bs num_segs) num_frames_per_seg hw d', num_frames_per_seg=num_frames_per_seg) # [bs*num_segs, num_frames_per_seg, 256, 1408] 

        # Aadptive Pooling for segment features
        frame_shape = (int(math.sqrt(segment_features.shape[2])), int(math.sqrt(segment_features.shape[2]))) # [16, 16] 
        hidden_states = einops.rearrange(segment_features, 'bs_num_segs num_frames_per_seg (h w) d -> bs_num_segs d num_frames_per_seg h w', h=frame_shape[0]) # [bs*num_segs, 1408, num_frames_per_seg, 16, 16]
        pool_size = 4
        hidden_states = nn.AdaptiveAvgPool3d([num_frames_per_seg, pool_size, pool_size])(hidden_states) # [bs*num_segs, 1408, num_frames_per_seg, 4, 4]  
        segment_features = einops.rearrange(hidden_states, '(bs num_segs) d num_frames_per_seg h w -> bs num_segs num_frames_per_seg (h w) d', num_segs=num_segs) # [bs, num_segs, num_frames_per_seg, 16, 1408]
        segment_features = einops.rearrange(segment_features, 'bs num_segs num_frames_per_seg hw d -> bs num_segs (num_frames_per_seg hw) d') # [bs, num_segs, num_frames_per_seg*16, 1408]


        """
        video features
        """
        if self.llm == 'llama3' or self.llm == 'vicuna':
            segment_features = self.video_projecter(segment_features) # [bs, num_segs, num_frames_per_seg*16, 4096]
            image_newline = self.image_newline[None, None, None, :].expand(batch_size, num_segs, 1, self.config.hidden_size).to(self.device)
        elif self.llm == 'phi3.5':
            segment_features = self.video_projecter(segment_features) # [bs, num_segs, num_frames_per_seg*16, 3072]
            image_newline = self.glb_GN[0,0,:][None, None, None, :].expand(batch_size, num_segs, 1, 4096).to(self.device)
            image_newline = self.multi_modal_projector(image_newline) # [bs, num_segs, 1, 3072]

        video_features = torch.cat([image_features, segment_features, image_newline], dim=2).to(self.device) # [bs, num_segs, 64+128+1, 4096]
        video_features = einops.rearrange(video_features, 'bs num_segs seq_len d -> bs (num_segs seq_len) d')

        return video_features

    def prepare_multimodal_inputs(self, batch_input_ids, batch_labels, batch_attention_mask, batch_image_features, batch_image_ids):
        new_input_embeds = []
        new_labels = []
        new_attention_masks = []
        for image_embeds, input_ids, labels, attention_mask, image_ids in zip(batch_image_features, batch_input_ids, batch_labels, batch_attention_mask, batch_image_ids):
            """
            image_embeds: [576, 4096]
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
            batch_image_features = self.encode_images(samples)
            inputs_embeds, labels, attention_masks = self.prepare_multimodal_inputs(batch_input_ids, batch_labels, batch_attention_mask, batch_image_features, samples['video_ids'])

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

        with self.maybe_autocast():
            # image_features
            batch_image_features = self.encode_images(samples)

            inputs_embeds, labels, attention_masks = self.prepare_multimodal_inputs(batch_input_ids, batch_labels, batch_attention_mask, batch_image_features, samples['video_ids'])

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










