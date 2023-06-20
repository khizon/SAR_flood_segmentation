import warnings
warnings.filterwarnings('ignore')

import os
import torch
from pytorch_lightning import LightningModule, Trainer, LightningDataModule, Callback, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from etci_dataset import ETCIDataset, ETCIDataModule
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from model import SegModule
from utils import *
import segmentation_models_pytorch as smp
import wandb
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='SAR Flood Segmentation')
    # where dataset will be stored
    parser.add_argument("--path", type=str, default="../data/")
    
    # Model
    parser.add_argument('--model', type=str, default='u-net')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2')
    parser.add_argument('--loss', type=str, default='dice')
    parser.add_argument('--pre_trained', default='no')
    
    #Model Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: )')
    parser.add_argument('--max_epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # 16-bit fp model to reduce the size
    parser.add_argument("--precision", default=16)
    parser.add_argument("--accelerator", default='auto')
    parser.add_argument("--devices", default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
    parser.add_argument("--early_stop", action=argparse.BooleanOptionalAction)
    parser.add_argument("--transforms", action=argparse.BooleanOptionalAction)   
    
    args = parser.parse_args()
    
    for k in args.__dict__:
        if args.__dict__[k] is None:
            args.__dict__[k] = True
            
    if 'segformer' in args.__dict__['model']:
        args.__dict__['backbone'] = None
        args.__dict__['pre_trained'] = 'ade-512-512'

    return args

class LogPredictionsCallback(Callback):
            
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Called when the test batch ends."""
        if batch_idx==0:
            self.log_table(batch, outputs, set='test')
           
    def log_table(self, batch, outputs, set='val'):
        x, y, water = batch['image'], batch['mask'], batch['water']
        y_hat = outputs['y_hat']

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

def create_model(args):
    
    model_class = {
        "u-net": smp.Unet,
        "u-net++": smp.UnetPlusPlus,
        "ma-net": smp.MAnet,
        "deeplabv3+": smp.DeepLabV3Plus,
        "fpn": smp.FPN
    }
    
    if args.model in model_class.keys():
        model = model_class[args.model](
            encoder_name= args.backbone,
            encoder_weights= args.pre_trained if args.pre_trained != 'no' else None ,
            in_channels=3,
            classes=1
        )
        
    elif 'segformer' in args.model:
        id2label = {'0': 'flood'}
        label2id = {v:k for k, v in id2label.items()}
        model_weights = f'nvidia/{args.model}-finetuned-{args.pre_trained}'

        model = SegformerForSemanticSegmentation.from_pretrained(model_weights, ignore_mismatched_sizes=True,
                                                                num_labels=len(id2label), id2label=id2label, label2id=label2id,
                                                                reshape_last_stage=True)
    
    return model

if __name__ == '__main__':
    os.chdir('src')
    args = get_args()
    print(args)
    print(os.getcwd())
    print(f'Pytorch {torch.__version__}')
    seed_everything(42, workers=True)
    
    datamodule=ETCIDataModule(args.path, batch_size=args.batch_size, num_workers=args.num_workers,
                              debug=args.debug, transforms=args.transforms)
    datamodule.setup()
    
#     if args.model == 'u-net':
#         model = smp.Unet(
#             encoder_name= args.backbone,
#             encoder_weights= args.pre_trained if args.pre_trained != 'no' else None ,
#             in_channels=3,
#             classes=1
#         )
#     elif args.model == 'u-net++':
#         model = smp.UnetPlusPlus(
#             encoder_name= args.backbone,
#             encoder_weights= args.pre_trained if args.pre_trained != 'no' else None ,
#             in_channels=3,
#             classes=1
#         )
        
#     elif args.model == 'ma-net':
#         model = smp.MAnet(
#             encoder_name= args.backbone,
#             encoder_weights= args.pre_trained if args.pre_trained != 'no' else None ,
#             in_channels=3,
#             classes=1
#         )
#     elif args.model == 'deeplabv3+':
#         model = smp.DeepLabV3Plus(
#             encoder_name= args.backbone,
#             encoder_weights= args.pre_trained if args.pre_trained != 'no' else None ,
#             in_channels=3,
#             classes=1
#         )
#     elif args.model == 'fpn':
#         model = smp.FPN(
#             encoder_name= args.backbone,
#             encoder_weights= args.pre_trained if args.pre_trained != 'no' else None ,
#             in_channels=3,
#             classes=1
#         )
    model = create_model(args)
    
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
                                            min_delta=0.25, patience=10, verbose=False, mode="max")
        callbacks.append(early_stop_callback)
    
    # Define Total Model
    model = SegModule(model, model_class=args.model, lr=args.lr, max_epochs=args.max_epochs, dropout=args.dropout, loss=args.loss)
    
    args.total_params = sum(
            param.numel() for param in model.parameters()
        )
    args.trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    
    wandb_logger = WandbLogger(project='sar_seg', log_model='all', config=vars(args))
    
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else args.accelerator,
                      devices=args.devices,
                      precision=args.precision,
                      max_epochs=args.max_epochs,
                      logger=wandb_logger if args.wandb else None,
                      gradient_clip_val=0.5,
                      callbacks=callbacks)
    
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path='best')
    
    wandb.finish()
    
    