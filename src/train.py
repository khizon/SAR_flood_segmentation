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
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='SAR Flood Segmentation')
    # where dataset will be stored
    parser.add_argument("--path", type=str, default="../data/")
    
    #Model Hyperparameters
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: )')
    parser.add_argument('--max_epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0)')
    # 16-bit fp model to reduce the size
    parser.add_argument("--precision", default=16)
    parser.add_argument("--accelerator", default='auto')
    parser.add_argument("--devices", default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
    parser.add_argument("--early_stop", action=argparse.BooleanOptionalAction)
    
    
    args = parser.parse_args()
    
    for k in args.__dict__:
        if args.__dict__[k] is None:
            args.__dict__[k] = True

    return args

class LogPredictionsCallback(Callback):
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Called when the validation batch ends."""
        if batch_idx == 0:
            self.log_table(batch, outputs, set='val')
            
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Called when the test batch ends."""
        if batch_idx == 0:
            self.log_table(batch, outputs, set='test')
            
            
            
    def log_table(self, batch, outputs, set='val'):
        x, y, water = batch['image'], batch['mask'], batch['water']
        y_hat = outputs['y_hat']
        miou = outputs[f'{set}_miou']

        n = x.size(dim=0)

        x = torch.squeeze(x, dim=1)
        x = torch.permute(x, (0,2,3,1))
        vv_images = [img[:,:,0].cpu().numpy() for img in x[:n]]
        vh_images = [img[:,:,1].cpu().numpy() for img in x[:n]]

        y = torch.squeeze(y, dim=1)
        labels = [img.cpu().numpy() for img in y[:n]]

        y_hat[y_hat >= 0.5] = 1
        y_hat[y_hat < 0.5] = 0
        preds = [img.cpu().numpy() for img in y_hat[:n]]

        water = torch.squeeze(water, dim=1)
        water_labels = [img.cpu().numpy() for img in water[:n]]

        columns = ['vv', 'vh', 'labels', 'predictions', 'water']
        data = [[wandb.Image(vv_i), wandb.Image(vh_i), wandb.Image(y_i), wandb.Image(yhat_i), wandb.Image(w_i)]
                    for vv_i, vh_i, y_i, yhat_i, w_i in list(zip(vv_images, vh_images, labels, preds, water_labels))]
        wandb_logger.log_table(key=f'SAR flood detection-{set}', columns=columns, data=data)

if __name__ == '__main__':
    os.chdir('src')
    args = get_args()
    print(args)
    print(os.getcwd())
    print(f'Pytorch {torch.__version__}')
    seed_everything(42, workers=True)
    
    datamodule=ETCIDataModule(args.path, batch_size=args.batch_size, num_workers=args.num_workers, debug=args.debug)
    datamodule.setup()
    
    model = smp.Unet(
        encoder_name='efficientnet-b7',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    )
    
    wandb_logger = WandbLogger(project='sar_seg', log_model='all')
    
    callbacks = []
    model_checkpoint = ModelCheckpoint(
            dirpath=os.path.join("checkpoints"),
            filename="sar-best-miou",
            save_top_k=1,
            verbose=True,
            monitor='val_miou',
            mode='max',
        )
    callbacks.append(model_checkpoint)
    log_predictions_callback = LogPredictionsCallback()
    callbacks.append(log_predictions_callback)
    
    if args.early_stop:
        early_stop_callback = EarlyStopping(monitor="val_miou",
                                            min_delta=0.25, patience=5, verbose=False, mode="max")
        callbacks.append(early_stop_callback)
    
    
    
    
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else args.accelerator,
                      devices=args.devices,
                      precision=args.precision,
                      max_epochs=args.max_epochs,
                      logger=wandb_logger if args.wandb else None,
                      gradient_clip_val=0.5,
                      callbacks=callbacks,
                      deterministic=True)
    
    unet_model = SegModule(model)
    
    trainer.fit(unet_model, datamodule=datamodule)
    trainer.test(unet_model, datamodule=datamodule, ckpt_path='best')
    
    wandb.finish()
    
    