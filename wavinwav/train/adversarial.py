from typing import Tuple
import torch
from torch import nn, Tensor
import torchaudio
from wavinwav.config import ModelConfig
from einops import rearrange

def get_2d_padding(kernel_size: Tuple[int, int], dilation: Tuple[int, int] = (1, 1)):
    return ((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2

class DiscriminatorBlock(nn.Module):
    def __init__(
            self,
            filters:int,
            in_channels:int,
            out_channels:int,
            n_fft:int,
            hop_length:int,
            win_length:int,
            kernel_size = (3,9),
            stride = (1,2),
            dilations = (1,2,4),
            normalized:bool = True,
            filters_scale = 1,
            max_filters = 1024
    ):
        super().__init__()

        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, window_fn=torch.hann_window,
            normalized=normalized, center=False,  power=None)
        spec_channels = 2 * in_channels

        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Conv2d(spec_channels, filters, kernel_size = kernel_size, padding = get_2d_padding(kernel_size))
        )
        in_chs = min(filters_scale * filters, max_filters)

        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i+1)) * self.filters, max_filters)
            self.convs.append(
                nn.Conv2d(in_chs, out_chs, kernel_size = kernel_size,
                          stride = stride, dilation = (dilation,1), padding= get_2d_padding(kernel_size, (dilation, 1)))
            )
            in_chs = out_chs
        out_chs = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
        self.convs.append(nn.Conv2d(in_chs, out_chs, kernel_size=(kernel_size[0], kernel_size[0]),
                                     padding=get_2d_padding((kernel_size[0], kernel_size[0]))))
        self.conv_post = nn.Conv2d(out_chs, out_channels,
                                    kernel_size=(kernel_size[0], kernel_size[0]),
                                    padding=get_2d_padding((kernel_size[0], kernel_size[0])))

    def forward(self, x:torch.Tensor):
        fmap = []
        z = self.spec_transform(x)
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, 'b c w t -> b c t w')
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):

    def __init__(self,config:ModelConfig):
        super().__init__()

        n_ffts = config.fft_sizes
        hop_lengths = config.hop_sizes
        win_lengths = config.win_lengths
        filters = config.num_filters
        in_channels = 1
        out_channels = 1
        sep_channels = False
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.sep_channels = sep_channels
        self.discriminators = nn.ModuleList([
            DiscriminatorBlock(filters, in_channels=in_channels, out_channels=out_channels,
                              n_fft=n_ffts[i], win_length=win_lengths[i], hop_length=hop_lengths[i])
            for i in range(len(n_ffts))
        ])

    @property
    def num_discriminators(self):
        return len(self.discriminators)

    def _separate_channels(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        return x.view(-1, 1, T)

    def forward(self, x: torch.Tensor):
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps