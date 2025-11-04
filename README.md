## Audio Hiding with Invertible Neural Networks

This repository provides an unofficial PyTorch implementation of the _[WavInWav: Time-domain Speech Hiding via Invertible Neural Network](https://www.arxiv.org/pdf/2510.02915)_ paper.

**Data hiding** involves embedding secret audio within a cover signal such that the concealed information remains imperceptible to human listeners. Since human hearing only spans roughly 20 Hz–20 kHz, much of an audio signal’s information lies in regions that can be modified without perceptual impact. An interesting read that partly talks about this property of continuous signals can be seen [here](https://sander.ai/2025/04/15/latents.html). This redundancy makes signals like images, video and audio a possible carrier for hidden data.

Most prior neural approaches perform hiding in the frequency domain. They transform the cover and secret signals using the Short-Time Fourier Transform (STFT), embed the secret into the magnitude spectrum, and reconstruct the waveform with the inverse STFT. However, this introduces two critical issues:

1. Lossy reconstruction: STFT-iSTFT introduces distortion.
2. Phase ambiguity: Frequency-domain methods discard phase information during STFT conversion, forcing the network to re-learn or approximate it, which degrades the recovered secret quality.

The authors of WavInWav address these issues by operating directly in the time domain using an invertible neural network (INN).

This research has exciting potential applications in audio watermarking, where users could not only distinguish AI-generated audio but also embed information about the audio’s origin or authenticity.

### Notes on this Implementation
1. The original paper optimizes a weighted sum of reconstruction losses between:
   * cover ↔ stego
   * original cover ↔ recovered cover
   * original secret ↔ recovered secret  
   To build on this, i've added optional support for additional objectives:
   * Distilation loss: l2 loss between [wavlm embeddings](https://huggingface.co/microsoft/wavlm-large) for each audio pair.
   * Adversarial loss: Using a Multi-Scale STFT discriminator from Facebook's [audiocraft](https://github.com/facebookresearch/audiocraft/tree/main/audiocraft/adversarial) library to improve perceptual realism.
2. I do not implement the Security Mechanism (section 3D) in the paper.
2. This implementation uses Pytorch Lightning for fast iteration.




### Quick Start:
```commandline
git clone https://github.com/odunola499/wavinwav.git
cd wavinwav
pip install -e .
```

Kick off a training job like so.
 ```python
 from wavinwav.train import get_loader, start_train
from wavinwav.config import HFDataConfig, TrainConfig, ModelConfig
from wavinwav.model import WavModel

train_data_config = HFDataConfig(
    hf_url = 'parler-tts/mls_eng_10k',
    hf_name = None,
    hf_split = 'train',
    audio_col_name = 'audio',
    sample_rate = 24000
)

valid_data_config = HFDataConfig(
    hf_url = 'parler-tts/mls_eng_10k',
    hf_name = None,
    hf_split = 'dev',
    audio_col_name = 'audio',
    sample_rate = 24000
)

train_config = TrainConfig()
model_config = ModelConfig()

model = WavModel(model_config)

train_loader = get_loader(train_data_config)
valid_loader = get_loader(valid_data_config)

start_train(model, train_config, model_config, train_loader, valid_loader)
 ```
For Adversarial training, 
 ```python
 from wavinwav.train import get_loader, start_train
from wavinwav.config import HFDataConfig, TrainConfig, ModelConfig
from wavinwav.model import WavModel

train_data_config = HFDataConfig(
    hf_url = 'parler-tts/mls_eng_10k',
    hf_name = None,
    hf_split = 'train',
    audio_col_name = 'audio',
    sample_rate = 24000
)

valid_data_config = HFDataConfig(
    hf_url = 'parler-tts/mls_eng_10k',
    hf_name = None,
    hf_split = 'dev',
    audio_col_name = 'audio',
    sample_rate = 24000
)

train_config = TrainConfig()
model_config = ModelConfig()

model = WavModel(model_config)
train_config.train_type = 'adversarial'


train_loader = get_loader(train_data_config)
valid_loader = get_loader(valid_data_config)

start_train(model, train_config, model_config, train_loader, valid_loader)
 ```

It may be safer to first do regular reconstruction loss training up to a point, then initialize this checkpoint for adversarial training.
 ```python
 from wavinwav.train import get_loader, start_train
from wavinwav.config import HFDataConfig, TrainConfig, ModelConfig
from wavinwav.model import WavModel

train_data_config = HFDataConfig(
    hf_url = 'parler-tts/mls_eng_10k',
    hf_name = None,
    hf_split = 'train',
    audio_col_name = 'audio',
    sample_rate = 24000
)

valid_data_config = HFDataConfig(
    hf_url = 'parler-tts/mls_eng_10k',
    hf_name = None,
    hf_split = 'dev',
    audio_col_name = 'audio',
    sample_rate = 24000
)

train_config = TrainConfig()
model_config = ModelConfig()

train_config.train_type = 'adversarial'
ckpt_path = 'checkpoint.ckpt'

model = WavModel(model_config)

train_loader = get_loader(train_data_config)
valid_loader = get_loader(valid_data_config)

start_train(model, train_config, model_config, train_loader, valid_loader, ckpt_path = ckpt_path)
 ```

### Credits
1. The Authors of the original paper
2. HuggingFace for their datasets library.
3. Lightning AI for their opensource work.