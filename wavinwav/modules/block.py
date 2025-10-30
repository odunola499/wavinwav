import math
import torch
from torch import nn, Tensor
from .conv import ResidualUnit, Snake1d, WNConv1d, WNConvTranspose1d


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=64, kernel_size=3):
        super().__init__()

        self.strides = [3, 3, 7, 7]
        enc_layers = []
        dec_layers = []

        in_ch = in_channels
        channel_stack = []
        for i, stride in enumerate(self.strides):
            out_ch = growth_rate * (i + 1)
            enc_layers += [
                ResidualUnit(in_ch, kernel=kernel_size, dilation=1, groups=1),
                ResidualUnit(in_ch, kernel=kernel_size, dilation=3, groups=1),
                ResidualUnit(in_ch, kernel=kernel_size, dilation=9, groups=1),
                Snake1d(in_ch),
                WNConv1d(
                    in_ch,
                    out_ch,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                )
            ]
            channel_stack.append((in_ch, out_ch))
            in_ch = out_ch

        for stride, (enc_in, enc_out) in zip(reversed(self.strides), reversed(channel_stack)):
            dec_layers += [
                Snake1d(enc_out),
                WNConvTranspose1d(
                    enc_out,
                    enc_in,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                    output_padding=stride % 2,
                ),
                ResidualUnit(enc_in, dilation=1, groups=1),
                ResidualUnit(enc_in, dilation=3, groups=1),
                ResidualUnit(enc_in, dilation=9, groups=1),
            ]

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AffineBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, stride, num_convs = 5, alpha = 0.1):
        super().__init__()
        self.phi = DenseBlock(
            in_channels,
            growth_rate,
            kernel_size,
        )

        self.rho = DenseBlock(
            in_channels,
            growth_rate,
            kernel_size,
        )

        self.eta = DenseBlock(
            in_channels,
            growth_rate,
            kernel_size,
        )
        self.alpha = alpha

    def forward(self, x_cover:Tensor, x_secret:Tensor):
        output_cover = x_cover + self.phi(x_secret)

        rho_out = self.alpha * self.rho(x_cover)
        eta_out = self.eta(output_cover)

        output_secret = (x_secret * torch.exp(rho_out)) + eta_out
        return output_cover, output_secret

    def inverse(self, x_stego:Tensor, z:Tensor):
        eta_out = self.eta(x_stego)
        rho_out = -self.alpha * self.rho(x_stego)

        z_out = (z - eta_out) * torch.exp(rho_out)
        x_out_stego = x_stego - self.phi(z_out)
        return x_out_stego, z_out
