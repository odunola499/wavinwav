import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint


class DenseBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 growth_rate = 64,
                 kernel_size = 3,
                 stride = 1,
                 num_convs = 5,
                 use_checkpoint = True
                 ):
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.ModuleList()
        self.use_checkpoint = use_checkpoint

        for i in range(num_convs - 1):
            in_ch = in_channels + (i * growth_rate)
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels= in_ch,
                        out_channels=growth_rate,
                        kernel_size=kernel_size,
                        padding = padding,
                        stride = stride,
                        groups=in_ch
                    ),
                    nn.LeakyReLU(inplace=True)
                )
            )
        final_in_ch = in_channels + ((num_convs -1) * growth_rate)
        self.final_conv = nn.Conv1d(
            in_channels = final_in_ch, out_channels= in_channels, kernel_size=1, stride = 1, padding = padding
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            inputs = torch.cat(features, dim = 1)
            if self.use_checkpoint and self.training:
                layer_out = checkpoint(layer, inputs, use_reentrant=False)
            else:
                layer_out = layer(inputs)
            features.append(layer_out)
        final_output = self.final_conv(torch.cat(features, dim = 1))
        return final_output

class AffineBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, stride, num_convs = 5, alpha = 0.1):
        super().__init__()
        self.phi = DenseBlock(
            in_channels,
            growth_rate,
            kernel_size,
            stride,
            num_convs
        )

        self.rho = DenseBlock(
            in_channels,
            growth_rate,
            kernel_size,
            stride,
            num_convs
        )

        self.eta = DenseBlock(
            in_channels,
            growth_rate,
            kernel_size,
            stride,
            num_convs
        )

        self.alpha = alpha

    def forward(self, x_cover:Tensor, x_secret:Tensor):
        output_cover = x_cover + self.phi(x_secret)

        rho_out = self.alpha * self.rho(x_cover)
        eta_out = self.eta(output_cover)

        output_secret = (x_secret * torch.exp(rho_out)) + eta_out
        return output_cover, output_secret

    def inverse(self, x_stego: Tensor, z: Tensor):
        eta_out = self.eta(x_stego)
        rho_out = -self.alpha * self.rho(x_stego)

        z_out = (z - eta_out) * torch.exp(rho_out)
        x_out_stego = x_stego - self.phi(z_out)
        return x_out_stego, z_out
