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