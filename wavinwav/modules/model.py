import torch
from torch import nn, Tensor
from .block import ForwardAffineBlock, InverseAffineBlock
from wavinwav.train.loss import *
from wavinwav.config import ModelConfig


class WavModel(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()
        num_invertible_blocks = config.num_inverible_blocks

        forward_blocks = nn.ModuleList([
            ForwardAffineBlock(
                in_channels=config.in_channels,
                growth_rate=config.growth_rate,
                kernel_size=config.kernel_size,
                stride = config.stride,
                num_convs=config.num_convs
            ) for _ in range(num_invertible_blocks)
        ])
        inverse_blocks = nn.ModuleList([
            InverseAffineBlock(
                in_channels=config.in_channels,
                growth_rate=config.growth_rate,
                kernel_size=config.kernel_size,
                stride=config.stride,
                num_convs=config.num_convs
            ) for _ in range(num_invertible_blocks)
        ])

        self.forward_blocks = forward_blocks
        self.inverse_blocks = inverse_blocks

        if config.tie_weights:
            self.tie_weights()
        self.config = config
        
    def tie_weights(self):
        for forward_block, inverse_block in zip(self.forward_blocks, self.inverse_blocks):
            for (forward_name, forward_param), (inverse_name, inverse_param) in zip(forward_block.named_parameters(), inverse_block.named_parameters()):
                inverse_param.data = forward_param.data


    def forward(self, x_cover:Tensor, x_secret:Tensor):
        for layer in self.forward_blocks:
            x_cover, x_secret = layer(x_cover, x_secret)

        x_stego = x_cover
        r = x_secret
        return x_stego, r

    def inverse(self, x_stego):
        z = torch.rand_like(x_stego)
        for layer in reversed(self.inverse_blocks):
            x_stego, z = layer(x_stego, z)

        x_cover = x_stego
        x_secret = z
        return x_cover, x_secret