import torch
from wavinwav.config import ModelConfig
from wavinwav.modules.model import WavModel

x_cover = torch.randn(4, 24000*10)
x_secret = torch.randn(4, 24000*10)

config = ModelConfig()
model = WavModel(config)

x_stego, r = model(x_cover, x_secret)
x_recon_cover, x_recon_secret = model.inverse(x_stego)
num_params = sum([p.numel() for p in model.parameters()])

print(f"Number of Parameters: {num_params}")
print(x_stego.shape)
print(x_recon_cover.shape)
print(x_recon_secret.shape)