import torch
from torch import nn, Tensor
from wavinwav.modules.block import AffineBlock
from wavinwav.train.loss import *
from wavinwav.config import ModelConfig


class WavModel(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()
        num_invertible_blocks = config.num_inverible_blocks

        blocks = nn.ModuleList([
            AffineBlock(
                in_channels=config.in_channels,
                growth_rate=config.growth_rate,
                kernel_size=config.kernel_size,
                stride=config.stride,
                num_convs=config.num_convs
            ) for _ in range(num_invertible_blocks)
        ])

        self.blocks = blocks
        self.config = config

    def forward(self, x_cover:Tensor, x_secret:Tensor):
        for layer in self.blocks:
            x_cover, x_secret = layer(x_cover, x_secret)

        x_stego = x_cover
        r = x_secret
        return x_stego, r

    def inverse(self, x_stego):
        z = torch.rand_like(x_stego)
        for layer in reversed(self.blocks):
            x_stego, z = layer.inverse(x_stego, z)

        x_cover = x_stego
        x_secret = z
        return x_cover, x_secret