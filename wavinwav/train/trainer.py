from datetime import datetime
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from lightning import LightningModule, Trainer
from wavinwav.config import TrainConfig, ModelConfig
from wavinwav.train.loss import ConcealingLoss, RevealingLoss
from wavinwav.train.adversarial import MultiScaleSTFTDiscriminator
from transformers import get_cosine_schedule_with_warmup
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger


class STFTTrainModule(LightningModule):
    def __init__(
            self,
            model:nn.Module,
            train_config:TrainConfig,
            model_config:ModelConfig,
            train_loader:DataLoader,
            valid_loader:DataLoader
    ):
        super().__init__()
        self.model = model
        self.train_config = train_config
        self.model_config = model_config
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.concealing_loss = ConcealingLoss(model_config)
        self.revealing_loss = RevealingLoss(model_config)

    def _internal_forward(self, x_cover:Tensor, x_secret:Tensor):
        x_stego, r = self.model(x_cover, x_secret)
        x_recovered_cover, x_recovered_secret = self.inverse(x_stego)

        stego_loss, z_loss = self.concealing_loss(x_cover, x_stego, r)
        cover_loss, secret_loss = self.revealing_loss(x_cover, x_recovered_cover,x_secret, x_recovered_secret)
        return stego_loss, z_loss, cover_loss, secret_loss

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def training_step(self, batch, batch_idx):
        x_cover = batch['x_cover']
        x_secret = batch['x_secret']
        stego_loss, z_loss, cover_loss, secret_loss = self._internal_forward(x_cover, x_secret)

        loss = stego_loss + z_loss + cover_loss + secret_loss
        self.log_dict({
            'train/stego_loss': stego_loss,
            'train/z_loss': z_loss,
            'train/cover_losss': cover_loss,
            'train/secret_loss':secret_loss
        })
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = self.train_config.lr,
            weight_decay=1e-2,
            fused = True
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.train_config.num_steps*0.1,
            num_training_steps=self.train_config.num_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


    def validation_step(self, batch, batch_idx):
        x_cover = batch['x_cover']
        x_secret = batch['x_secret']
        stego_loss, z_loss, cover_loss, secret_loss = self._internal_forward(x_cover, x_secret)

        loss = stego_loss + z_loss + cover_loss + secret_loss
        self.log_dict({
            'validation/stego_loss': stego_loss,
            'validation/z_loss': z_loss,
            'validation/cover_loss': cover_loss,
            'validation/secret_loss': secret_loss
        })
        return loss


class AdversarialTrainModule(LightningModule):
    def __init__(
            self,model:nn.Module,
            train_config:TrainConfig,
            model_config:ModelConfig,
            train_loader: DataLoader,
            valid_loader: DataLoader,

    ):
        super().__init__()
        self.model = model
        self.train_config = train_config
        self.model_config = model_config
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.concealing_loss = ConcealingLoss(model_config)
        self.revealing_loss = RevealingLoss(model_config)

        self.discriminator = MultiScaleSTFTDiscriminator(model_config)
        self.automatic_optimization = False

    def configure_optimizers(self):
        gen_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = self.train_config.lr,
            weight_decay=1e-2,
            fused = True
        )
        gen_scheduler = get_cosine_schedule_with_warmup(
            gen_optimizer,
            num_warmup_steps=self.train_config.num_steps*0.1,
            num_training_steps=self.train_config.num_steps
        )

        disc_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.train_config.lr,
            weight_decay=1e-2,
            fused=True
        )
        disc_scheduler = get_cosine_schedule_with_warmup(
            disc_optimizer,
            num_warmup_steps=self.train_config.num_steps * 0.1,
            num_training_steps=self.train_config.num_steps
        )

        return [gen_optimizer, disc_optimizer], [{"scheduler": gen_scheduler, "interval": "step"},{"scheduler": disc_scheduler, "interval": "step"} ]

    def training_step(self, batch, batch_idx):
        x_cover = batch['x_cover']
        x_secret = batch['x_secret']

        stego, r, x_recovered_cover, x_recovered_secret = self.generator_forward(x_cover, x_secret)
        gen_optimizer, disc_optimizer = self.optimizers()
        gen_scheduler, disc_scheduler = self.lr_schedulers()

        disc_optimizer.zero_grad()
        real_disc_inputs = torch.concat([x_cover, x_secret], dim = 0)
        real_pred = self.discriminator(real_disc_inputs)
        loss_real = F.mse_loss(real_pred, torch.ones_like(real_pred))

        fake_disc_inputs = torch.concat([stego, x_recovered_cover,x_recovered_secret], dim = 0)
        fake_pred = self.discriminator(fake_disc_inputs)
        loss_fake = F.mse_loss(fake_pred,torch.zeros_like(fake_pred))

        disc_loss = 0.5 * (loss_real + loss_fake)
        self.manual_backward(disc_loss)
        disc_optimizer.step()
        disc_scheduler.step()

        gen_optimizer.zero_grad()
        stego, r, x_recovered_cover, x_recovered_secret = self.generator_forward(x_cover, x_secret)
        gen_output = torch.concat([stego, x_recovered_cover, x_recovered_secret], dim = 0)
        with torch.no_grad():
            fake_pred = self.discriminator(gen_output)
        gen_loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))

        stego_loss, z_loss = self.concealing_loss(x_cover, stego, r)
        cover_loss, secret_loss = self.revealing_loss(x_cover, x_recovered_cover, x_secret, x_recovered_secret)
        gen_loss += (stego_loss + z_loss + cover_loss + secret_loss)

        self.manual_backward(0.5 * gen_loss)
        gen_optimizer.step()
        gen_scheduler.step()

        self.log_dict({
            'train/disc_loss': disc_loss,
            'train/gen_loss': gen_loss,
            'train/stego_loss': stego_loss,
            'train/z_loss': z_loss,
            'train/cover_loss': cover_loss,
            'train/secret_loss': secret_loss
        })

    def validation_step(self, batch, batch_idx):
        x_cover = batch['x_cover']
        x_secret = batch['x_secret']

        stego, r, x_recovered_cover, x_recovered_secret = self.generator_forward(x_cover, x_secret)

        real_disc_inputs = torch.concat([x_cover, x_secret], dim=0)
        real_pred = self.discriminator(real_disc_inputs)
        loss_real = F.mse_loss(real_pred, torch.ones_like(real_pred))

        fake_disc_inputs = torch.concat([stego, x_recovered_cover, x_recovered_secret], dim=0)
        fake_pred = self.discriminator(fake_disc_inputs)
        loss_fake = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
        disc_loss = 0.5 * (loss_real + loss_fake)

        stego, r, x_recovered_cover, x_recovered_secret = self.generator_forward(x_cover, x_secret)
        gen_output = torch.concat([stego, x_recovered_cover, x_recovered_secret], dim=0)
        with torch.no_grad():
            fake_pred = self.discriminator(gen_output)
        gen_loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))

        stego_loss, z_loss = self.concealing_loss(x_cover, stego, r)
        cover_loss, secret_loss = self.revealing_loss(x_cover, x_recovered_cover, x_secret, x_recovered_secret)
        gen_loss += (stego_loss + z_loss + cover_loss + secret_loss)

        self.log_dict({
            'valid/disc_loss': disc_loss,
            'valid/gen_loss': gen_loss,
            'valid/stego_loss': stego_loss,
            'valid/z_loss': z_loss,
            'valid/cover_loss': cover_loss,
            'valid/secret_loss': secret_loss
        })

    def generator_forward(self, x_cover:Tensor, x_secret:Tensor):
        x_stego, r = self.model(x_cover, x_secret)
        x_recovered_cover, x_recovered_secret = self.model.inverse(x_stego)
        return x_stego, r, x_recovered_cover, x_recovered_secret



def start_train(
        model:nn.Module,
        train_config:TrainConfig,
        model_config:ModelConfig,
        train_loader:DataLoader,
        valid_loader:DataLoader,
        ckpt_path = None
):
    train_type = train_config.train_type

    if ckpt_path is not None:
        print(f"Loading weights from {ckpt_path}")
        weights = torch.load(ckpt_path)['state_dict']
        weights = {i[6:]:j for i,j in weights.items()}
        model.load_state_dict(weights)

    if train_type == 'stft':
        print("Using STFTTrainModule")
        module = STFTTrainModule(
            model = model,
            train_config = train_config,
            model_config = model_config,
            train_loader = train_loader,
            valid_loader = valid_loader
        )

    elif train_type == 'adversarial':
        print("Using AdversarialTrainModule")
        module = AdversarialTrainModule(
            model=model,
            train_config=train_config,
            model_config=model_config,
            train_loader=train_loader,
            valid_loader=valid_loader
        )
    else:
        raise ValueError(f"{train_type} is not  supported")

    run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger = WandbLogger(
        project = 'WAVINWAV', name = run_name
    )
    checkpoint = ModelCheckpoint(
        monitor='val/loss',
        dirpath='final_checkpoints',
        filename=f'model-{{epoch:02d}}-{{step:02d}}-{{val/loss:.3f}}',
        save_top_k=1,
        mode='min',
        every_n_train_steps=1000,
        save_last=True,
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        device = 'auto',
        precision = "bf16-mixed",
        max_steps = train_config.num_steps,
        accumulate_grad_batches=train_config.accumulate_grad_batch,
        log_every_n_steps=train_config.log_every_n_steps,
        logger = logger,
        callbacks = [lr_monitor, checkpoint],
        limit_val_batches = train_config.limit_val_batches
    )

    trainer.fit(module)



