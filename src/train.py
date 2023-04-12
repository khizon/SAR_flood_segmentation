import warnings
warnings.filterwarnings('ignore')

import os
import torch
from pytorch_lightning import LightningModule, Trainer, LightningDataModule, Callback, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from etci_dataset import ETCIDataset, ETCIDataModule
from model import SegModule
from utils import *
import segmentation_models_pytorch as smp
import wandb


if __name__ == '__main__':
    os.chdir('src')
    print(os.getcwd())
    print(torch.__version__)
    seed_everything(42, workers=True)
    
    datamodule=ETCIDataModule('../data/', batch_size=8, num_workers=1, debug=True)
    datamodule.setup()
    
    model = smp.Unet(
        encoder_name='efficientnet-b7',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    )
    
    wandb_logger = WandbLogger(project='sar_seg', log_model='all')
    model_checkpoint = ModelCheckpoint(
            dirpath=os.path.join("checkpoints"),
            filename="sar-best-miou",
            save_top_k=1,
            verbose=True,
            monitor='val_miou',
            mode='max',
        )
    
    early_stop_callback = EarlyStopping(monitor="val_miou",
                                        min_delta=0.25, patience=5, verbose=False, mode="max")
    
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'auto',
                      devices=1,
                      precision=16,
                      max_epochs=30,
                      logger=wandb_logger,
                      gradient_clip_val=0.5,
                      callbacks=[model_checkpoint, early_stop_callback],
                      deterministic=True)
    
    unet_model = SegModule(model)
    
    trainer.fit(unet_model, datamodule=datamodule)
    trainer.test(unet_model, datamodule=datamodule, ckpt_path='best')
    
    wandb.finish()
    
    