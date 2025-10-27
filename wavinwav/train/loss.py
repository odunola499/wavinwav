import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from wavinwav.config import Config

class STFTLoss(nn.Module):
    def __init__(self, config:Config):
        super().__init__()

        self.fft_sizes = config.fft_sizes
        self.hop_sizes = config.hop_sizes
        self.win_lengths = config.win_lengths

        for i in config.win_lengths:
            self.register_buffer(f"window_{i}", torch.hann_window(i))

    @torch.no_grad()
    def stft(self, x, fft_size, hop_size, win_length):
        window = getattr(self, f"window_{win_length}")
        stft = torch.stft(
            x, n_fft=fft_size, hop_length=hop_size, win_length = win_length, window=window, return_complex=True
        )
        mag = torch.abs(stft)
        return mag

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor):
        total_loss = 0.0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            S_pred = self.stft(x_pred, fft_size, hop_size, win_length)
            S_target = self.stft(x_target, fft_size, hop_size, win_length)

            sc_loss = torch.norm(S_target - S_pred, p='fro') / (torch.norm(S_target, p='fro') + 1e-9)
            mag_loss = F.l1_loss(torch.log(S_target + 1e-7), torch.log(S_pred + 1e-7))

            total_loss += (sc_loss + mag_loss)

        return total_loss / len(self.fft_sizes)

class ConcealingLoss(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.stft_func = STFTLoss(config)

    def stego_loss(self, x_cover:Tensor, x_stego:Tensor):
        return self.stft_func(x_cover, x_stego)

    def z_loss(self,r:Tensor):
        output = r**2 + torch.log(torch.tensor(2 * math.pi))
        return 0.5 * output

    def forward(self, x_cover:Tensor, x_stego:Tensor, r:Tensor):
        stego_loss = self.stego_loss(x_cover, x_stego)
        stego_loss = self.config.stego_coef * stego_loss

        z_loss = self.z_loss(r)
        z_loss = self.config.z_coef * z_loss
        return stego_loss, z_loss

class RevealingLoss(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.stft_func = STFTLoss(config)

    def cover_loss(self, x_cover:Tensor, x_rev_cover:Tensor):
        return self.stft_func(x_cover, x_rev_cover)

    def secret_loss(self, x_orig_secret, x_restored_secret):
        return self.stft_func(x_orig_secret, x_restored_secret)

    def forward(self, x_cover:Tensor, x_rev_cover:Tensor, x_orig_secret:Tensor, x_restored_secret:Tensor):
        cover_loss = self.cover_loss(x_cover, x_rev_cover)
        cover_loss = self.config.cover_coef * cover_loss

        secret_loss = self.secret_loss(x_orig_secret, x_restored_secret)
        secret_loss = self.config.secret_coef * secret_loss
        return cover_loss, secret_loss

class PerceptualLoss(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        pass