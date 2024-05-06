# Adapted from https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/models/controlnet.py

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class PoseGuider(ModelMixin, ConfigMixin):
    """Pose guider model for the control network.

    Quoting from https://arxiv.org/abs/2311.17117: "This Pose Guider utilizes four convolution
    layers (4×4 kernels, 2×2 strides, using 16,32,64,128 channels, similar to the condition encoder
    in [https://arxiv.org/abs/2302.05543]) to align the pose image with the same resolution as the
    noise latent."

    Differences from the original implementation:
    - Channels are (16, 32, 96, 256) instead of (16, 32, 64, 128)
    - Conditioning embedding channels set to align with noise latent after pre-processing

    Parameters:
        conditioning_embedding_channels (int): Number of channels in the output embedding.
        conditioning_channels (int): Number of channels in the input conditioning tensor.
        block_out_channels (Tuple[int, ...]): Number of channels in each block of the network.
    """

    @register_to_config
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2)
            )

        self.conv_out = zero_module(
            nn.Conv2d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding
