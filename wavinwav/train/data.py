import io
import random
import torch
from datasets import load_dataset, Audio
import torchaudio
from torch.utils.data import Dataset, DataLoader
from wavinwav.config import HFDataConfig

class HFData(Dataset):
    def __init__(self, data, sample_rate, audio_col_name:str = 'audio'):
        self.data = data
        self.col_name = audio_col_name
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def load_audio(self, audio_bytes):
        audio, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        return audio

    def __getitem__(self, index):
        cover_row = self.data[index]
        secret_row = self.data[random.randrange(0, len(self.data))]

        x_cover = cover_row[self.col_name]['bytes']
        x_secret = secret_row[self.col_name]['bytes']

        x_cover = self.load_audio(x_cover)
        x_secret = self.load_audio(x_secret)

        return {
            'cover':x_cover,
            'secret':x_secret
        }

def collate_fn(batch):
    x_cover = [i['cover'] for i in batch]
    x_secret = [i['secret'] for i in batch]

    x_cover = torch.stack(x_cover)
    x_secret = torch.stack(x_secret)
    return {
        'x_cover':x_cover,
        'x_secret': x_secret
    }

def get_loader(config:HFDataConfig, batch_size = 4):
    data = load_dataset(config.hf_url, config.hf_name, split = config.hf_split)
    data = data.cast_column(config.audio_col_name, Audio(decode = False))

    dataset = HFData(data, sample_rate = config.sample_rate, audio_col_name= config.audio_col_name)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn=collate_fn)
    return loader


