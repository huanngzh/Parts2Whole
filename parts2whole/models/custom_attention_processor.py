from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
    SpatialNorm,
)
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.torch_utils import maybe_allow_in_graph
from einops import rearrange


def default_set_attn_proc_func(
    name: str,
    hidden_size: int,
    cross_attention_dim: Optional[int],
    ori_attn_proc: object,
) -> object:
    return ori_attn_proc


def set_unet_2d_condition_attn_processor(
    unet: UNet2DConditionModel,
    set_self_attn_proc_func: Callable = default_set_attn_proc_func,
    set_cross_attn_proc_func: Callable = default_set_attn_proc_func,
) -> None:
    attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None or "motion_modules" in name:
            # self attention
            attn_procs[name] = set_self_attn_proc_func(
                name, hidden_size, cross_attention_dim, attn_processor
            )
        else:
            # cross attention
            attn_procs[name] = set_cross_attn_proc_func(
                name, hidden_size, cross_attention_dim, attn_processor
            )
    unet.set_attn_processor(attn_procs)


class DecoupledCrossAttnProcessor2_0(nn.Module):
    r"""
    Attention processor for decoupled cross attention for PyTorch 2.0.
    """

    def __init__(
        self, hidden_size, cross_attention_dim=None, max_image_length=6, scale=1.0
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.max_image_length = max_image_length
        self.scale = scale

        self.to_k_dc = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )
        self.to_v_dc = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        # split hidden states
        end_pos = encoder_hidden_states.shape[1] - self.max_image_length
        encoder_hidden_states, dc_hidden_states = (
            encoder_hidden_states[:, :end_pos, :],
            encoder_hidden_states[:, end_pos:, :],
        )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # for decoupled cross attention
        dc_key = self.to_k_dc(dc_hidden_states)
        dc_value = self.to_v_dc(dc_hidden_states)

        dc_key = dc_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        dc_value = dc_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        dc_hidden_states = F.scaled_dot_product_attention(
            query, dc_key, dc_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        dc_hidden_states = dc_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        dc_hidden_states = dc_hidden_states.to(query.dtype)

        hidden_states = hidden_states + self.scale * dc_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
