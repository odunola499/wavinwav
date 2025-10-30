import torch
from wavinwav.config import ModelConfig
from wavinwav.model import WavModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_cover = torch.randn(4, 1, 24000*10, device = device)
x_secret = torch.randn(4,1,  24000*10, device = device)

config = ModelConfig()
model = WavModel(config).to(device)

x_stego, r = model(x_cover, x_secret)
x_recon_cover, x_recon_secret = model.inverse(x_stego)
num_params = sum([p.numel() for p in model.parameters()])

print(f"Number of Parameters: {num_params}")
print(x_stego.shape)
print(x_recon_cover.shape)
print(x_recon_secret.shape)