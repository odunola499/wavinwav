import torch
from torch import nn
from wavinwav.config import ModelConfig


class Discriminator(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()
        pass