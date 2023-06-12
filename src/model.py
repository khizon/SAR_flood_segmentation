from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryF1Score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp
import torch
import numpy as np
import wandb

def seed_everything(seed=2**3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

class CombinedLoss(torch.nn.Module):
    def __init__(self, mode="binary", gamma=2, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = smp.losses.DiceLoss(mode=mode)
        self.focal_loss = smp.losses.FocalLoss(mode=mode, gamma=gamma, alpha=alpha)

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        combined_loss = (dice_loss + focal_loss) / 2.0
        return combined_loss

class SegModule(LightningModule):
    def __init__(self, model, lr=1e-3, max_epochs=30, dropout=0.1, loss='dice', **kwargs):
        
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        if loss == 'dice':
            self.loss = smp.losses.DiceLoss(mode="binary")
        elif loss == 'BCE':
            self.loss = smp.losses.SoftBCEWithLogitsLoss()
        elif loss == 'focal':
            self.loss = smp.losses.FocalLoss(mode="binary")
        elif loss == 'combined':
            self.loss = CombinedLoss()
            
        
        self.jaccard_f, self.jaccard_b = BinaryJaccardIndex(ignore_index=0), BinaryJaccardIndex(ignore_index=1)
        self.jaccard_m, self.precision, self.recall, self.f1 = BinaryJaccardIndex(), BinaryPrecision(), BinaryRecall(), BinaryF1Score()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.train_step_outputs = []
        self.max_epochs=max_epochs
        self.lr = lr
        self.dropout=torch.nn.Dropout(dropout)

        
    def forward(self, x):
        y_hat = self.model(x)
        y_hat = self.dropout(y_hat)
        return y_hat
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask'].unsqueeze(dim=1)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.train_step_outputs.append({
            'train_loss':loss
        })
        return {'loss': loss}
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["train_loss"] for x in self.train_step_outputs]).mean()
        self.log("train_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.train_step_outputs.clear()
        

    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask'].unsqueeze(dim=1)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        # miou = self.metric(y_hat, y.int())
        self.test_step_outputs.append({
            'test_miou': self.jaccard_m(y_hat, y.int()),
            'test_iou_f': self.jaccard_f(y_hat, y.int()),
            'test_iou_b': self.jaccard_b(y_hat, y.int()),
            'test_precision':self.precision(y_hat, y.int()),
            'test_recall':self.recall(y_hat, y.int()),
            'test_f1':self.f1(y_hat, y.int()),
            'test_loss':loss
        })
        return {"y_hat": y_hat, "test_loss": loss}

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x["test_loss"] for x in self.test_step_outputs]).mean()
        avg_miou = torch.stack([x["test_miou"] for x in self.test_step_outputs]).mean()
        avg_iou_f = torch.stack([x["test_iou_f"] for x in self.test_step_outputs]).mean()
        avg_iou_b = torch.stack([x["test_iou_b"] for x in self.test_step_outputs]).mean()
        avg_precision = torch.stack([x["test_precision"] for x in self.test_step_outputs]).mean()
        avg_recall = torch.stack([x["test_recall"] for x in self.test_step_outputs]).mean()
        avg_f1 = torch.stack([x["test_f1"] for x in self.test_step_outputs]).mean()
        
        self.log("test_loss", avg_loss, on_epoch=True, prog_bar=False)
        self.log("test_miou", avg_miou*100., on_epoch=True, prog_bar=True)
        self.log("test_iou_f", avg_iou_f*100., on_epoch=True, prog_bar=False)
        self.log("test_iou_b", avg_iou_b*100., on_epoch=True, prog_bar=False)
        self.log("test_precision", avg_precision*100., on_epoch=True, prog_bar=False)
        self.log("test_recall", avg_recall*100., on_epoch=True, prog_bar=False)
        self.log("test_f1", avg_f1*100., on_epoch=True, prog_bar=False)
        self.test_step_outputs.clear()
        
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask'].unsqueeze(dim=1)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        # miou = self.metric(y_hat, y.int())
        self.validation_step_outputs.append({
            'val_miou': self.jaccard_m(y_hat, y.int()),
            'val_iou_f': self.jaccard_f(y_hat, y.int()),
            'val_iou_b': self.jaccard_b(y_hat, y.int()),
            'val_precision':self.precision(y_hat, y.int()),
            'val_recall':self.recall(y_hat, y.int()),
            'val_f1':self.f1(y_hat, y.int()),
            'val_loss':loss
        })
        return {"y_hat": y_hat, "val_loss": loss}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        avg_miou = torch.stack([x["val_miou"] for x in self.validation_step_outputs]).mean()
        avg_iou_f = torch.stack([x["val_iou_f"] for x in self.validation_step_outputs]).mean()
        avg_iou_b = torch.stack([x["val_iou_b"] for x in self.validation_step_outputs]).mean()
        avg_precision = torch.stack([x["val_precision"] for x in self.validation_step_outputs]).mean()
        avg_recall = torch.stack([x["val_recall"] for x in self.validation_step_outputs]).mean()
        avg_f1 = torch.stack([x["val_f1"] for x in self.validation_step_outputs]).mean()
        
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=False)
        self.log("val_miou", avg_miou*100., on_epoch=True, prog_bar=True)
        self.log("val_iou_f", avg_iou_f*100., on_epoch=True, prog_bar=False)
        self.log("val_iou_b", avg_iou_b*100., on_epoch=True, prog_bar=False)
        self.log("val_precision", avg_precision*100., on_epoch=True, prog_bar=False)
        self.log("val_recall", avg_recall*100., on_epoch=True, prog_bar=False)
        self.log("val_f1", avg_f1*100., on_epoch=True, prog_bar=False)
        self.validation_step_outputs.clear()
        
        

    