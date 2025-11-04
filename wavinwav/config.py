from dataclasses import dataclass
from typing import List, Literal, Optional

@dataclass
class HFDataConfig:
    hf_url: str = 'parler-tts/mls_eng_10k'
    hf_name: Optional[str] = None
    hf_split: str = 'train'
    audio_col_name:str = 'audio'
    sample_rate:int = 24000

@dataclass
class ModelConfig:
    in_channels:int =  1
    growth_rate:int = 16
    kernel_size:int = 3
    stride:int = 1
    num_convs = 5
    alpha:int = 0.1
    num_invertible_blocks = 5
    tie_weights:bool = True
    add_noise_layer:bool = False
    sample_rate:int= 24000

    stego_coef:int = 1
    z_coef:int = 1
    cover_coef:int = 1
    secret_coef:int = 1
    use_stft:bool = True
    encoders:List[Literal['whisper', 'wavlm']] = ('whisper','wavlm')
    noise_pool:List[str] = ("identity", "speckle", "gaussian", "resample", "clean")
    fft_sizes:list = (512, 1024, 2048)
    hop_sizes:list = (128, 256, 512)
    win_lengths:list = (512, 1024, 2048)
    num_filters:int = 32

@dataclass
class TrainConfig:
    ckpt_dir:int = 'checkpoints'
    num_steps:int = 50000
    lr:float = 3e-4
    precision:str = 'bf16-mixed'
    accumulate_grad_batch:int = 4
    val_check_interval:int = 2000
    log_every_n_steps:int = 4
    limit_val_batches:int = 50
    sample_rate:int = 24000
    train_type: Literal['adversarial', 'stft'] = 'stft'
    log_to_wandb:bool = True

