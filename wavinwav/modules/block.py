import math
import torch
from torch import nn, Tensor
from torch.nn.utils.parametrizations import weight_norm

def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1, kernel=7, groups=1):
        super().__init__()
        pad = ((kernel - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=kernel, dilation=dilation, padding=pad, groups=groups),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y

class Block(nn.Module):
    def __init__(self, in_channels, growth_rate = 64, kernel_size = 3, stride = 3):
        super().__init__()

        up_block = nn.Sequential(
            ResidualUnit(in_channels, kernel = kernel_size,dilation=1, groups=1),
            ResidualUnit(in_channels,kernel = kernel_size, dilation=3, groups=1),
            ResidualUnit(in_channels, kernel = kernel_size, dilation=9, groups=1),
            Snake1d(in_channels),
            WNConv1d(
                in_channels,
                growth_rate,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )
        down_block = nn.Sequential(
            Snake1d(growth_rate),
            WNConvTranspose1d(
                growth_rate,
                in_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            ResidualUnit(in_channels, dilation=1, groups=1),
            ResidualUnit(in_channels, dilation=3, groups=1),
            ResidualUnit(in_channels, dilation=9, groups=1),
        )
        self.up_blocks = up_block
        self.down_blocks = down_block

    def forward(self,x):
        x = self.up_blocks(x)
        x = self.down_blocks(x)
        return x



class DenseBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 growth_rate = 64,
                 kernel_size = 3,
                 stride = 1,
                 num_convs = 5):
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.ModuleList()

        for i in range(num_convs - 1):
            in_ch = in_channels + (i * growth_rate)
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels= in_ch,
                        out_channels=growth_rate,
                        kernel_size=kernel_size,
                        padding = padding,
                        stride = stride
                    ),
                    nn.LeakyReLU(inplace=True)
                )
            )
        final_in_ch = in_channels + ((num_convs -1) * growth_rate)
        self.final_conv = nn.Conv1d(
            in_channels = final_in_ch, out_channels= in_channels, kernel_size=kernel_size, stride = 1, padding = padding
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            inputs = torch.cat(features, dim = 1)
            layer_out = layer(inputs)
            features.append(layer_out)
        final_output = self.final_conv(torch.cat(features, dim = 1))
        return final_output


class AffineBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, stride, num_convs = 5, alpha = 0.1):
        super().__init__()
        self.phi = Block(
            in_channels,
            growth_rate,
            kernel_size,
            stride,
        )

        self.rho = Block(
            in_channels,
            growth_rate,
            kernel_size,
            stride,
        )

        self.eta = Block(
            in_channels,
            growth_rate,
            kernel_size,
            stride,
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
