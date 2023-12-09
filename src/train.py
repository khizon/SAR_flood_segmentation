import warnings
warnings.filterwarnings('ignore')

import os
import torch
import torchvision.models.segmentation as tms
from pytorch_lightning import LightningModule, Trainer, LightningDataModule, Callback, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
# from etci_dataset import ETCIDataset, ETCIDataModule
from sen1floods11_dataset import Sen1Floods11Dataset, Sen1Floods11DataModule
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from model import SegModule
from utils import *
import segmentation_models_pytorch as smp
import wandb
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='SAR Flood Segmentation')
    # where dataset will be stored
    parser.add_argument("--path", type=str, default="sen1floods11")
    parser.add_argument("--label_type", type=str, default="HandLabeled")
    
    # Model
    parser.add_argument('--model', type=str, default='u-net')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2')
    parser.add_argument('--loss', type=str, default='dice')
    parser.add_argument('--pre_trained', default='no')
    
    #Model Hyperparameters
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
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
    
    # Early Stopping
    parser.add_argument("--delta", default=0.01)
    parser.add_argument("--patience", default=20)
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
        class_labels = {-1: "N/A", 0: "Not Flood", 1: "Flood"}
        x, y = batch['img'], batch['label'],
        y_hat = outputs['y_hat']

        n = x.size(dim=0)

        x = torch.squeeze(x, dim=1)
        x = torch.permute(x, (0,2,3,1))
        vv_images = [img[:,:,0].cpu().numpy() for img in x[:n]]
#         vh_images = [img[:,:,1].cpu().numpy() for img in x[:n]]

        y = torch.squeeze(y, dim=1)
        y[y==-1] = 255
        labels = [img.cpu().numpy() for img in y[:n]]

        # y_hat[y_hat >= 0.5] = 1
        # y_hat[y_hat < 0.5] = 0
        y_hat = torch.squeeze(y_hat, dim=1)
        preds = [img.cpu().float().numpy() for img in y_hat[:n]]

        columns = ['annotations']
        data =[[
            wandb.Image(
                vv_i, masks={
                    "label": {"mask_data": y_i, "class_labels": class_labels},
                    "prediction": {"mask_data": yhat_i, "class_labels": class_labels}
                }
            ),
        ] for vv_i, y_i, yhat_i in list(zip(vv_images, labels, preds))
        ]
        wandb_logger.log_table(key=f'SAR Flood Detection-{set}', columns=columns, data=data)

def create_model(args):
    
    model_class = {
        "u-net": smp.Unet,
        "u-net++": smp.UnetPlusPlus,
        "ma-net": smp.MAnet,
        "deeplabv3+": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
    }
    image_processor = None
    
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
        # Freeze encoder
        # for param in model.segformer.encoder.parameters():
        #     param.requires_grad = False
        image_processor = AutoImageProcessor.from_pretrained(model_weights)
    
    return model, image_processor

if __name__ == '__main__':
    ROOT = os.getcwd()
    args = get_args()
    print(args)
    print(f'Current Working Directory: {ROOT}')
    print(f'Pytorch {torch.__version__}')
    seed_everything(42, workers=True)
    
    if args.label_type == 'HandLabeled':
        path = os.path.join(ROOT, args.path, 'hand_labeled.csv')
    elif args.label_type == 'WeaklyLabeled':
        path = os.path.join(ROOT, args.path, 'weak_labeled.csv')

    model, image_processor = create_model(args)

    # datamodule=ETCIDataModule(args.path, batch_size=args.batch_size, num_workers=args.num_workers,
    #                           debug=args.debug, transforms=args.transforms)
    print(f'CSV location:{path}')
    datamodule = Sen1Floods11DataModule(path, args.label_type, batch_size=args.batch_size, num_workers=args.num_workers,
                              debug=args.debug, transforms=args.transforms)
    datamodule.setup()
    
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
                                            min_delta=args.delta, patience=args.patience, verbose=False, mode="max")
        callbacks.append(early_stop_callback)
    
    # Define Total Model
    model = SegModule(model, model_class=args.model, lr=args.lr, max_epochs=args.max_epochs, dropout=args.dropout, loss=args.loss)
    
    args.total_params = sum(
            param.numel() for param in model.parameters()
        )
    args.trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    
    wandb_logger = WandbLogger(project='sar_seg_sen1floods11', log_model='all', config=vars(args))
    
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
    
    # WandB cleanup
    dry_run = False
    api = wandb.Api(overrides={"project": "sar_seg_sen1floods11", "entity": "khizon"})
    project = api.project('sar_seg_sen1floods11')


    for artifact_type in project.artifacts_types():
        for artifact_collection in artifact_type.collections():
            for version in api.artifact_versions(artifact_type.type, artifact_collection.name):
                if artifact_type.type == 'model':
                    if len(version.aliases) > 0:
                        # print out the name of the one we are keeping
                        print(f'KEEPING {version.name}')
                    else:
                        print(f'DELETING {version.name}')
                        if not dry_run:
                            version.delete()
    
    