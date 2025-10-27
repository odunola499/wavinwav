from dataclasses import dataclass

@dataclass
class Config:
    in_channels:int =  1 # Mono, 2 for stereo
    growth_rate:int = 64
    kernel_size:int = 3
    stride:int = 1
    num_convs = 5
    alpha:int = 0.1
    num_inverible_blocks = 5
    tie_weights:bool = True
    add_noise_layer:bool = False
    stego_coef:int = 1
    z_coef:int = 1
    cover_coef:int = 1
    secret_coef:int = 1
    fft_sizes:list = (512, 1024, 2048)
    hop_sizes:list = (128, 256, 512)
    win_lengths:list = (512, 1024, 2048)