import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, MultivariateNormal


def seed_everything(seed:int = 42):
    torch.manual_seed(seed)


class AffineCoupling(nn.Module):
    def __init__(self, dim:int, hidden:int, mask:torch.Tensor):
        super().__init__()
        self.dim = dim
        self.register_buffer('mask', mask)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace = True),
            nn.Linear(hidden, dim * 2)
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x:torch.Tensor):
        x_a = x * self.mask
        x_b = x * (1- self.mask)

        st = self.net(x_a)
        s, t = st.chunk(2, dim = 1)
        s = torch.tanh(s) * 2.0

        y_a= x_a
        y_b = x_b * torch.exp(s) + t

        y = y_a + y_b

        log_det = ((1 - self.mask) * s).sm(dim = 1)
        return y, log_det

    def inverse(self, y:torch.Tensor):
        y_a = y * self.mask
        y_b = y * (1- self.mask)

        st = self.net(y_a)
        s, t = st.chunk(2,dim = 1)
        s = torch.tanh(s) * 2.0

        x_a = y_a
        x_b = (y_b - t) * (torch.exp(-s))
        x = x_a + x_b

        log_det = ((1- self.mask) * s).sum(dim = 1)
        return x, log_det


class Flip(nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        self.dim = dim

    def forward(self, x:torch.Tensor):
        return torch.flip(x, dims = [1]), x.new_zeros(x.shape[0])

    def inverse(self, y:torch.Tensor):
        return torch.flip(y, dims = [1]), y.new_zeros(y.shape[0])

class NVP(nn.Module):
    def __init__(self, dim:int = 2, hidden:int = 128, n_layers:int = 6):
        super().__init__()
        self.dim = dim
        layers = []
        for i in range(n_layers):
            mask = self._make_mask(i % 2)
            layers.append(AffineCoupling(dim, hidden, mask))
            layers.append(Flip(dim))
        self.layers = nn.ModuleList(layers)

        self.register_buffer('base_mean', torch.zeros(dim))
        self.register_buffer('base_logstd', torch.zeros(dim))

    def _make_mask(self, parity:int):
        mask = torch.zeros(1, self.dim)
        half = self.dim // 2
        if parity == 1:
            mask[:half] = 1.0
        else:
            mask[half:] = 1.0
        return mask

    def forward(self, z:torch.Tensor):
        log_det_sum = z.new_zeros(z.shape[0])
        x = z
        for layer in self.layers:
            x, ld = layer(x)
            log_det_sum += ld
        return x, log_det_sum

    def inverse(self, x:torch.Tensor):
        log_det_sum = x.new_zeros(x.shape[0])
        z = x
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            log_det_sum += ld
        return z, log_det_sum

    def log_prob(self, x:torch.Tensor):
        z, log_det = self.inverse(x)
        std = torch.exp(self.base_logstd)
        base = Normal(self.base_mean, std)
        log_pz = base.log_prob(z).sum(dim = 1)
        return log_pz + log_det

    def sample(self, n:int = 1):
        z = torch.randn(n, self.dim, device = self.base_mean.device)
        x, _ = self.forward(z)
        return x