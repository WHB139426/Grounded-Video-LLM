"""
    Modified from https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2

    Copyright 2024 OpenGVLab/InternVideo
    Copyright 2024 MBZUAI ORYX
"""

import numpy as np
import torch
import logging
import math
import logging
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn

import torch.utils.checkpoint as checkpoint
from functools import partial
from einops import rearrange

logger = logging.getLogger(__name__)

# from flash_attn.modules.mlp import FusedMLP
# from flash_attn.ops.rms_norm import DropoutAddRMSNorm


# --------------------------------------------------------
# 3D sine-cosine position embedding
# References:
# MVD: https://github.com/ruiwang2021/mvd/blob/main/modeling_finetune.py
# --------------------------------------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(
        embed_dim_spatial, grid
    )

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(
        embed_dim_temporal, grid_t
    )

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_size ** 2, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed(checkpoint_model, model, orig_t_size=4, pos_name='vision_encoder.pos_embed'):
    if pos_name in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model[pos_name]
        embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
        num_patches = model.patch_embed.num_patches  #
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

        # we use 4 frames for pretraining
        new_t_size = model.T
        # height (== width) for the checkpoint position embedding
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (orig_t_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int((num_patches // (new_t_size)) ** 0.5)

        # class_token and dist_token are kept unchanged
        if orig_t_size != new_t_size:
            logger.info(f"Temporal interpolate from {orig_t_size} to {new_t_size} ({pos_name})")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> B， T, HW, C -> BHW, C, T  (B = 1)
            pos_tokens = pos_tokens.view(1, orig_t_size, -1, embedding_size)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=new_t_size, mode='linear')
            pos_tokens = pos_tokens.view(1, -1, embedding_size, new_t_size)
            pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_name] = new_pos_embed
            pos_embed_checkpoint = new_pos_embed

        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            logger.info(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size} ({pos_name})")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size)
            pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_name] = new_pos_embed


def interpolate_pos_embed_internvideo2(checkpoint_model, model, orig_t_size=8):
    # interpolate position embedding
    for pos_name in ['pos_embed', 'clip_pos_embed']:
        if pos_name in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model[pos_name]
            embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
            num_patches = model.patch_embed.num_patches  #
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

            # we use 8 frames for pretraining
            # new_t_size = args.num_frames * args.num_segments // videogpt_plus.patch_embed.tubelet_size
            new_t_size = model.num_frames // model.tubelet_size
            # height (== width) for the checkpoint position embedding
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (orig_t_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (new_t_size)) ** 0.5)

            # class_token and dist_token are kept unchanged
            if orig_t_size != new_t_size:
                logger.info(f"Temporal interpolate from {orig_t_size} to {new_t_size} ({pos_name})")
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> B， T, HW, C -> BHW, C, T  (B = 1)
                pos_tokens = pos_tokens.view(1, orig_t_size, -1, embedding_size)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
                pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=new_t_size, mode='linear')
                pos_tokens = pos_tokens.view(1, -1, embedding_size, new_t_size)
                pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model[pos_name] = new_pos_embed
                pos_embed_checkpoint = new_pos_embed

            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                logger.info(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size} ({pos_name})")
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model[pos_name] = new_pos_embed

    if 'pos_embed_spatial' in checkpoint_model or 'pos_embed_temporal' in checkpoint_model:
        raise NotImplementedError


def interpolate_pos_embed_internvideo2_new(checkpoint_model, model, orig_t_size=8):
    pos_names = []
    for k in checkpoint_model.keys():
        if ('pos_embed' in k or 'clip_pos_embed' in k) and 'img_pos_embed' not in k:
            pos_names.append(k)

    # logger.info(f"pos names list for interpolating: {pos_names}")

    assert len(pos_names) > 0, checkpoint_model.keys()

    if 'pos_embed_spatial' in checkpoint_model.keys() or 'pos_embed_temporal' in checkpoint_model.keys():
        raise NotImplementedError

    # interpolate position embedding
    for pos_name in pos_names:

        pos_embed_checkpoint = checkpoint_model[pos_name]
        embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
        num_patches = model.patch_embed.num_patches  #
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

        # we use 8 frames for pretraining
        # new_t_size = args.num_frames * args.num_segments // model.patch_embed.tubelet_size
        new_t_size = model.num_frames // model.tubelet_size
        # height (== width) for the checkpoint position embedding
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (orig_t_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int((num_patches // (new_t_size)) ** 0.5)

        # class_token and dist_token are kept unchanged
        if orig_t_size != new_t_size:
            logger.info(f"Temporal interpolate from {orig_t_size} to {new_t_size} ({pos_name})")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> B， T, HW, C -> BHW, C, T  (B = 1)
            pos_tokens = pos_tokens.view(1, orig_t_size, -1, embedding_size)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=new_t_size, mode='linear')
            pos_tokens = pos_tokens.view(1, -1, embedding_size, new_t_size)
            pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_name] = new_pos_embed
            pos_embed_checkpoint = new_pos_embed

        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            logger.info(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size} ({pos_name})")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size)
            pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_name] = new_pos_embed

















class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentiveBlock(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None, out_dim=None):
        super().__init__()

        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)

        return x


class AttentionPoolingBlock(AttentiveBlock):

    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False, force_fp32=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.force_fp32 = force_fp32

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if self.force_fp32:
            output_type = x.dtype
            out = x.float().mul_(self.gamma.float()) if self.inplace else x.float() * self.gamma.float()
            return out.to(dtype=output_type)
        else:
            out = x.mul_(self.gamma) if self.inplace else x * self.gamma
            return out

import torch
import torch.nn as nn

from einops import rearrange

from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                          device=qkv.device)
                output = flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                             indices, batch_size, seqlen),
                                   'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

        return output, None
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_flash_attn=False,
                 causal=False, norm_layer=nn.LayerNorm, qk_normalization=False, use_fused_rmsnorm=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.use_fused_rmsnorm = use_fused_rmsnorm

    def _naive_attn(self, x):
        B, N, C = x.shape
        # print(x.shape, torch.cuda.memory_allocated(), torch.cuda.memory_allocated())
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        # attn = attn - attn.max(-1)[0].unsqueeze(-1)  # in case of overflow for fp16
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # print(torch.cuda.memory_allocated(), torch.cuda.memory_allocated())
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):

        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            if self.use_fused_rmsnorm:
                q = self.q_norm(q.flatten(-2, -1))[0].view(q.shape)
                k = self.k_norm(k.flatten(-2, -1))[0].view(k.shape)
            else:
                q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal
        )
        outs = self.proj(rearrange(context, "b s h d -> b s (h d)"))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, x):
        x = self._naive_attn(x) if not self.use_flash_attn else self._flash_attn(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_flash_attn=False, use_fused_mlp=False,
            fused_mlp_heuristic=1, with_cp=False, qk_normalization=False, layerscale_no_force_fp32=False,
            use_fused_rmsnorm=False):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              use_flash_attn=use_flash_attn, causal=False, norm_layer=norm_layer,
                              qk_normalization=qk_normalization,
                              use_fused_rmsnorm=use_fused_rmsnorm)
        self.ls1 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_fused_mlp:
            self.mlp = FusedMLP(in_features=dim, hidden_features=mlp_hidden_dim, heuristic=fused_mlp_heuristic)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.with_cp = with_cp
        self.use_fused_rmsnorm = use_fused_rmsnorm

    def forward(self, x, residual=None):

        def _inner_forward(x, residual=None):
            if self.use_fused_rmsnorm:
                x, residual = self.norm1(x, residual)
                x = self.drop_path1(self.ls1(self.attn(x)))
                x, residual = self.norm2(x, residual)
                x = self.drop_path2(self.ls2(self.mlp(x)))
                return x, residual
            else:
                assert residual is None
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                return x

        if self.with_cp:
            # print(f"\033[31m use_checkpoint [0m")
            return checkpoint.checkpoint(_inner_forward, x, residual)
        else:
            return _inner_forward(x, residual=residual)


class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
            num_frames=8, tubelet_size=1, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1]
        )  # (T, H, W)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.num_img_patches = self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_channels=in_chans, out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1])
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(3).permute(0, 2, 3, 1)  # B x C x T x HW => B x T x HW x C
        x = self.norm(x)
        return x


class Linear_Decoder(nn.Module):
    def __init__(self, in_channels=1408, out_channels=3200,
                 norm_layer=nn.LayerNorm, clip_norm_type='l2'):
        super().__init__()
        self.clip_norm_type = clip_norm_type
        # logger.info(f'Normalization Type: {clip_norm_type}')

        self.head = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm(self.head(x))

        if self.clip_norm_type == 'l2':
            x = x / x.norm(dim=-1, keepdim=True)
        elif self.clip_norm_type == 'none':
            pass
        else:
            raise NotImplementedError

        return x


from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union
from functools import partial

class PretrainInternVideo2(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            patch_size: int = 14,
            img_size: int = 224,
            qkv_bias: bool = False,
            drop_path_rate: float = 0.25,
            embed_dim: int = 1408,
            num_heads: int = 16,
            mlp_ratio: float = 48 / 11,
            init_values: float = 1e-5,
            qk_normalization: bool = True,
            depth: int = 40,
            use_flash_attn: bool = True,
            use_fused_rmsnorm: bool = True,
            use_fused_mlp: bool = True,
            fused_mlp_heuristic: int = 1,
            attn_pool_num_heads: int = 16,
            clip_embed_dim: int = 768,
            layerscale_no_force_fp32: bool = False,
            num_frames: int = 8,
            tubelet_size: int = 1,
            sep_pos_embed: bool = False,
            sep_image_video_pos_embed: bool = False,
            use_checkpoint: bool = False,
            checkpoint_num: int = 0,
            # for unmasked teacher
            clip_teacher_embed_dim: int = 3200,
            clip_teacher_final_dim: int = 768,  # if 0, not distill final features
            clip_norm_type: str = 'l2',
            clip_return_layer: int = 1,
            clip_student_return_interval: int = 1,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        # assert use_flash_attn == use_fused_rmsnorm == use_fused_mlp, 'use_flash_attn, use_fused_rmsnorm and use_fused_mlp should be consistent'

        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim

        self.depth = depth
        self.clip_norm_type = clip_norm_type
        self.return_index = []
        for i in range(clip_return_layer):
            self.return_index.append(depth - int(i * clip_student_return_interval) - 1)
        # logger.info(f'Normalization Type: {clip_norm_type}')
        # logger.info(f'Strudent Return Index: {self.return_index}')

        if use_fused_rmsnorm:
            norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        else:
            norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
            
        norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        self.norm_layer_for_blocks = norm_layer_for_blocks
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim,
            num_frames=num_frames, tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        num_img_patches = self.patch_embed.num_img_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # stolen from https://github.com/facebookresearch/mae_st/blob/dc072aaaf640d06892e23a33b42223a994efe272/models_vit.py#L65-L73C17
        self.sep_pos_embed = sep_pos_embed
        self.sep_image_video_pos_embed = sep_image_video_pos_embed
        if sep_pos_embed:
            raise NotImplementedError
        else:
            if sep_image_video_pos_embed:
                # logger.info("Use joint position embedding, for image and video we use different pos_embed.")
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
                self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches + 1, embed_dim))
                # for CLIP decoder
                self.clip_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
                self.clip_img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches + 1, embed_dim))
            else:
                # logger.info("Use joint position embedding, for image and video we use same pos_embed.")
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
                self.clip_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # choose which layer to use checkpoint
        with_cp_list = [False] * depth
        if use_checkpoint:
            for idx in range(depth):
                if idx < checkpoint_num:
                    with_cp_list[idx] = True
        # logger.info(f"Droppath rate: {dpr}")
        # logger.info(f"Checkpoint list: {with_cp_list}")

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=norm_layer_for_blocks,
                  drop_path=dpr[i], init_values=init_values, attn_drop=0.,
                  use_flash_attn=use_flash_attn, use_fused_mlp=use_fused_mlp,
                  fused_mlp_heuristic=fused_mlp_heuristic,
                  with_cp=with_cp_list[i],
                  qk_normalization=qk_normalization,
                  layerscale_no_force_fp32=layerscale_no_force_fp32,
                  use_fused_rmsnorm=use_fused_rmsnorm)
            for i in range(depth)])
        self.clip_projector = AttentionPoolingBlock(
            dim=embed_dim, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim)

        # CLIP decoder
        self.clip_decoder = nn.ModuleList([
            Linear_Decoder(
                in_channels=embed_dim,
                out_channels=clip_teacher_embed_dim,
                norm_layer=partial(nn.LayerNorm, eps=1e-5),
                clip_norm_type=clip_norm_type
            ) for _ in range(clip_return_layer)
        ])
        self.final_clip_decoder = nn.Identity()
        if clip_teacher_final_dim > 0:
            self.final_clip_decoder = Linear_Decoder(
                in_channels=clip_embed_dim,
                out_channels=clip_teacher_final_dim,
                norm_layer=partial(nn.LayerNorm, eps=1e-5),
                clip_norm_type=clip_norm_type
            )

        self.init_pos_embed()
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then the _entire_ featurizer."""
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return transformer_block_policy

    def init_pos_embed(self):
        # logger.info("Init pos_embed from sincos pos_embed")
        if self.sep_pos_embed:
            raise NotImplementedError
        else:
            # trunc_normal_(self.pos_embed, std=.02)
            # trunc_normal_(self.clip_pos_embed, std=.02)
            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.patch_embed.grid_size[1],  # height & weight
                self.patch_embed.grid_size[0],  # t_size
                cls_token=True
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            self.clip_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            if self.sep_image_video_pos_embed:
                img_pos_embed = get_3d_sincos_pos_embed(
                    self.pos_embed.shape[-1],
                    self.patch_embed.grid_size[1],  # height & weight
                    1,
                    cls_token=True
                )
                self.img_pos_embed.data.copy_(torch.from_numpy(img_pos_embed).float().unsqueeze(0))
                self.clip_img_pos_embed.data.copy_(torch.from_numpy(img_pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed',
            'pos_embed_spatial',
            'pos_embed_temporal',
            'pos_embed_cls',
            'img_pos_embed',
            'cls_token',
            'clip_pos_embed',
            'clip_pos_embed_spatial',
            'clip_pos_embed_temporal',
            'clip_pos_embed_cls',
            'clip_img_pos_embed'
        }

    # @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, mask=None, use_image=False, x_vis_return_idx=-1, x_vis_only=False):

        x = self.patch_embed(x.type(self.dtype))

        # print(f"x.shape: {x.shape} x.dtype: {x.dtype}, videogpt_plus.dtype: {self.dtype}")
        B, T, L, C = x.shape  # T: temporal; L: spatial
        x = x.view([B, T * L, C])


        # append cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add pos_embed
        if self.sep_pos_embed:
            raise NotImplementedError
        else:
            if use_image:
                if self.sep_image_video_pos_embed:
                    pos_embed = self.img_pos_embed
                else:
                    # (1, num_img_patches + 1, embed_dim)
                    # print('origin pos_embed.shape:', self.pos_embed.shape)
                    cls_pos_embed = self.pos_embed[:, 0:1, :]
                    # print('cls_pos_embed.shape:', cls_pos_embed.shape)

                    img_pos_embed = self.pos_embed[:, 1:, :].view(1, self.num_frames,
                                                                  self.patch_embed.num_patches // self.num_frames,
                                                                  self.embed_dim).mean(dim=1)
                    # print('img_pos_embed.shape:', img_pos_embed.shape)

                    pos_embed = torch.cat([cls_pos_embed, img_pos_embed], dim=1)
                    # print('final img_pos_embed.shape:', pos_embed.shape)
            else:
                pos_embed = self.pos_embed
        x = x + pos_embed

        # mask tokens, ~mask means visible
        if mask is not None:
            x = x[~mask].reshape(B, -1, C)
        else:
            x = x.reshape(B, -1, C)

        residual = None
        x_clip = []
        for idx, blk in enumerate(self.blocks):
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            # print(f"\033[31m这是{idx}, {x.shape}\033[0m")
            x = blk(x, residual=residual)
            # return intermediate features
            if idx in self.return_index:
                if isinstance(x, tuple) and len(x) == 2:
                    tmp_x, tmp_residual = x
                    if residual is not None:
                        x_clip.append(tmp_x + tmp_residual)
                else:
                    x_clip.append(x)
            if idx == (self.depth + x_vis_return_idx):
                # print(f'idx = {idx} len(self.blocks)={len(self.blocks)}')
                break

        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual

        x_vis = x

        if x_vis_only:
            return x_vis

        x_pool_vis = self.clip_projector(x_vis)
        x_align = self.final_clip_decoder(x_pool_vis)



        # align CLIP
        x_clip = torch.stack(x_clip)
        K, B, _, C_CLIP = x_clip.shape
        # add pos_embed
        if self.sep_pos_embed:
            raise NotImplementedError
        else:
            if use_image:
                if self.sep_image_video_pos_embed:
                    clip_pos_embed = self.clip_img_pos_embed
                else:
                    # (1, num_img_patches + 1, embed_dim)
                    # print('origin pos_embed.shape:', self.pos_embed.shape)
                    clip_cls_pos_embed = self.clip_pos_embed[:, 0:1, :]
                    # print('cls_pos_embed.shape:', cls_pos_embed.shape)

                    clip_img_pos_embed = self.clip_pos_embed[:, 1:, :].view(1, self.num_frames,
                                                                            self.patch_embed.num_patches // self.num_frames,
                                                                            self.embed_dim).mean(dim=1)
                    # print('img_pos_embed.shape:', img_pos_embed.shape)

                    clip_pos_embed = torch.cat([clip_cls_pos_embed, clip_img_pos_embed], dim=1)
                    # print('final img_pos_embed.shape:', pos_embed.shape)

            else:
                clip_pos_embed = self.clip_pos_embed

        clip_pos_embed = clip_pos_embed.repeat(B, 1, 1)
        if mask is not None:
            x_clip = x_clip + clip_pos_embed[~mask].view(B, -1, C_CLIP).unsqueeze(0).repeat(K, 1, 1, 1)
        else:
            x_clip = x_clip + clip_pos_embed.view(B, -1, C_CLIP).unsqueeze(0).repeat(K, 1, 1, 1)

        # CLIP decoder
        x_clip_align = []
        for idx, clip_decoder in enumerate(self.clip_decoder):
            x_clip_align.append(clip_decoder(x_clip[idx]))
        x_clip_align = torch.stack(x_clip_align)

        return x_vis, x_pool_vis, x_clip_align, x_align


def pretrain_internvideo2_1b_patch14_224(num_frames, use_flash_attn=False):
    model = PretrainInternVideo2(
        in_chans=3, img_size=224, patch_size=14,
        embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48 / 11,
        clip_embed_dim=768,
        attn_pool_num_heads=16, qkv_bias=False,
        drop_path_rate=0.25,
        init_values=0.00001,
        qk_normalization=True,
        use_flash_attn=use_flash_attn,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
        fused_mlp_heuristic=1,
        layerscale_no_force_fp32=False,
        num_frames=num_frames,
        tubelet_size=1,
        sep_pos_embed=False,
        sep_image_video_pos_embed=True,
        use_checkpoint=False,
        checkpoint_num=40,
        clip_teacher_embed_dim=3200,
        clip_teacher_final_dim=768,
        clip_norm_type='l2',
        clip_return_layer=6,
        clip_student_return_interval=1,
    )

    return model
