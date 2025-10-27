from random import choice
import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.functional import resample

class NoiseLayer(nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate
        self.noise_pool = ["identity", "speckle", "gaussian", "resample", "clean"]

    def forward(self, x):
        noise = choice(self.noise_pool)
        if noise == 'identity':
            return x
        elif noise == 'speckle':
            noise = torch.randn_like(x)
            return x + x * noise * 0.1
        elif noise == 'gaussian':
            noise = torch.randn_like(x) * 0.01
            return x + noise
        elif noise == 'resample':
            new_sr = int(self.sample_rate * torch.empty(1).uniform_(0.8, 1.2))
            x = resample(x, self.sample_rate, new_sr)
            x = resample(x, new_sr, self.sample_rate)
            return x
        else:
            return x

