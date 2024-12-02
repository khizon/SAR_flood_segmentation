from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryF1Score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import segmentation_models_pytorch as smp
import torch
import numpy as np
import wandb
from transformers import AutoImageProcessor

def seed_everything(seed=2**3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

class FocalDiceLoss(torch.nn.Module):
    def __init__(self, mode="binary", gamma=2, alpha=0.5):
        super(FocalDiceLoss, self).__init__()
        self.dice_loss = smp.losses.DiceLoss(mode=mode)
        self.focal_loss = smp.losses.FocalLoss(mode=mode, gamma=gamma, alpha=alpha)

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        combined_loss = (dice_loss + focal_loss) / 2.0
        return combined_loss
    
class BCEDiceLoss(torch.nn.Module):
    def __init__(self, mode="binary"):
        super(BCEDiceLoss, self).__init__()
        self.dice_loss = smp.losses.DiceLoss(mode=mode)
        self.BCE_loss = smp.losses.SoftBCEWithLogitsLoss()

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.BCE_loss(pred, target)
        combined_loss = (dice_loss + bce_loss) / 2.0
        return combined_loss

class SegModule(LightningModule):
    def __init__(self, model, model_class='u-net', lr=1e-3, max_epochs=30, dropout=0.1,
                 loss='dice', debug=True, scheduler='CosineAnnealingLR', **kwargs):
        
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.model_class = model_class
        self.debug = debug
        self.scheduler = scheduler
        
        if loss == 'dice':
            self.loss = smp.losses.DiceLoss(mode="binary", ignore_index=-1)
        elif loss == 'BCE':
            self.loss = smp.losses.SoftBCEWithLogitsLoss(ignore_index=-1)
        elif loss == 'focal':
            self.loss = smp.losses.FocalLoss(mode="binary", ignore_index=-1)
        elif loss == 'Focal+Dice':
            self.loss = FocalDiceLoss()
        elif loss == 'BCE+Dice':
            self.loss = BCEDiceLoss()

        # self.jaccard_f, self.jaccard_b = BinaryJaccardIndex(ignore_index=0), BinaryJaccardIndex(ignore_index=1)
        self.jaccard_m, self.precision, self.recall, self.f1 = BinaryJaccardIndex(ignore_index=-1), BinaryPrecision(ignore_index=-1), BinaryRecall(ignore_index=-1), BinaryF1Score(ignore_index=-1)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.holdout_step_outputs = []
        self.train_step_outputs = []
        self.max_epochs=max_epochs
        self.lr = lr
        self.dropout=torch.nn.Dropout(dropout)

        
    def forward(self, x):
        if 'segformer' in self.model_class:
            outputs = self.model(pixel_values=x)
            # Upsample logits to 256 x 256
            y_hat = torch.nn.functional.interpolate(outputs.logits, size=(x.shape[2],x.shape[3]), mode="bilinear", align_corners=False)
        elif self.model_class == 'fcn':
            y_hat = self.model(x)['out']
        else:
            y_hat = self.model(x)
        if self.training:
            y_hat = self.dropout(y_hat)
        
        # Post-processing step
        # Create a mask for pixels with value 999 in the first two channels of x
        mask = (x[:, 0:2, :, :] == 999).any(dim=1, keepdim=True)
        # Use the mask to set corresponding pixels in y_hat to 0
        y_hat[mask] = -1e3
        
        return y_hat
 
    def configure_optimizers(self):
        optimizer = Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr)
        
        schedulers = {
            'CosineAnnealingLR': CosineAnnealingLR(optimizer, T_max=self.max_epochs),
            'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2,
                                                eta_min=1e-8, last_epoch=-1, verbose=False)
        }
        scheduler = schedulers[self.scheduler]
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch['img'], batch['label'].unsqueeze(dim=1)
        
        # Debug: Simulate NaN issue
        # if self.debug and (batch_idx==0) and (self.current_epoch==0):
        #     x = torch.full_like(x, np.nan)
        
        y_hat = self(x)
        # If y_hat or Loss contains NaNs, skip the batch.
        if torch.isnan(y_hat).any():
            print(f"Skipping Batch {batch_idx}: NaN detected on y_hat\n")
            return None
        
        loss = self.loss(y_hat, y)

        if torch.isnan(loss):
            print(f"Skipping Batch {batch_idx}: Loss is NaN\n")
            return None
        
        self.train_step_outputs.append({
            'train_loss': loss
        })

        return {'loss': loss}
        
    def on_train_epoch_end(self):
        avg_loss = torch.nanmean(torch.stack([x["train_loss"] for x in self.train_step_outputs]))
        self.log("train_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.train_step_outputs.clear()
        

    def test_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch['img'], batch['label'].unsqueeze(dim=1)
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        if dataloader_idx == 0:
            self.test_step_outputs.append({
                'test_miou': self.jaccard_m(y_hat, y.int()),
                'test_precision':self.precision(y_hat, y.int()),
                'test_recall':self.recall(y_hat, y.int()),
                'test_f1':self.f1(y_hat, y.int()),
                'test_loss':loss
            })
        else:
            self.holdout_step_outputs.append({
                'test_miou': self.jaccard_m(y_hat, y.int()),
                'test_precision':self.precision(y_hat, y.int()),
                'test_recall':self.recall(y_hat, y.int()),
                'test_f1':self.f1(y_hat, y.int()),
                'test_loss':loss
            })
        
        y_hat = torch.where(torch.sigmoid(y_hat) >= 0.5, 1, 0)
        return {"y_hat": y_hat, "test_loss": loss}

    def on_test_epoch_end(self):
        results = [self.test_step_outputs, self.holdout_step_outputs]
        
        for idx, result in enumerate(results):
            d_idx = {0: 'test', 1: 'holdout'}
            avg_loss = torch.nanmean(torch.stack([x["test_loss"] for x in result]))
            avg_miou = torch.nanmean(torch.stack([x["test_miou"] for x in result]))
            avg_precision = torch.nanmean(torch.stack([x["test_precision"] for x in result]))
            avg_recall = torch.nanmean(torch.stack([x["test_recall"] for x in result]))
            avg_f1 = torch.nanmean(torch.stack([x["test_f1"] for x in result]))

            self.log(f"{d_idx[idx]}_loss", avg_loss, on_epoch=True, prog_bar=False)
            self.log(f"{d_idx[idx]}_miou", avg_miou*100., on_epoch=True, prog_bar=True)
            self.log(f"{d_idx[idx]}_precision", avg_precision*100., on_epoch=True, prog_bar=False)
            self.log(f"{d_idx[idx]}_recall", avg_recall*100., on_epoch=True, prog_bar=False)
            self.log(f"{d_idx[idx]}_f1", avg_f1*100., on_epoch=True, prog_bar=False)
            result.clear()
        
    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['label'].unsqueeze(dim=1)
        y_hat = self(x)

        # If y_hat or Loss contains NaNs, skip the batch.
        if torch.isnan(y_hat).any():
            print(f"Validation {batch_idx}: NaN detected on y_hat\n")
        
        loss = self.loss(y_hat, y)

        if torch.isnan(loss):
            print(f"Validation Batch {batch_idx}: Loss is NaN\n")
        
        self.validation_step_outputs.append({
            'val_miou': self.jaccard_m(y_hat, y.int()),
            'val_precision':self.precision(y_hat, y.int()),
            'val_recall':self.recall(y_hat, y.int()),
            'val_f1':self.f1(y_hat, y.int()),
            'val_loss':loss
        })
        return {"y_hat": y_hat, "val_loss": loss}

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_loss = torch.nanmean(torch.stack([x["val_loss"] for x in self.validation_step_outputs]))
            avg_miou = torch.nanmean(torch.stack([x["val_miou"] for x in self.validation_step_outputs]))
            avg_precision = torch.nanmean(torch.stack([x["val_precision"] for x in self.validation_step_outputs]))
            avg_recall = torch.nanmean(torch.stack([x["val_recall"] for x in self.validation_step_outputs]))
            avg_f1 = torch.nanmean(torch.stack([x["val_f1"] for x in self.validation_step_outputs]))
            
            self.log("val_loss", avg_loss, on_epoch=True, prog_bar=False)
            self.log("val_miou", avg_miou*100., on_epoch=True, prog_bar=True)
            self.log("val_precision", avg_precision*100., on_epoch=True, prog_bar=False)
            self.log("val_recall", avg_recall*100., on_epoch=True, prog_bar=False)
            self.log("val_f1", avg_f1*100., on_epoch=True, prog_bar=False)
        else:
            print("No valid validation batches were processed this epoch.")
        
        self.validation_step_outputs.clear()

        
        

    