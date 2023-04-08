from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryJaccardIndex
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp
import torch
import numpy as np

def seed_everything(seed=2**3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SegModule(LightningModule):
    def __init__(self, model, lr=1e-3, max_epochs=30, **kwargs):
        
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.metric = BinaryJaccardIndex()
        self.validation_step_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask'].unsqueeze(dim=1)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return {'loss': loss}
    
    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask'].unsqueeze(dim=1)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        miou = self.metric(y_hat, y)
        return {"y_hat": y_hat, "test_loss": loss, "test_miou": miou}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_miou = torch.stack([x["test_miou"] for x in outputs]).mean()
        self.log("test_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("test_miou", avg_miou*100., on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask'].unsqueeze(dim=1)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        
        miou = self.metric(y_hat, y)
        self.validation_step_outputs.append({
            'val_miou':miou,
            'val_loss':loss
        })
        return {"val_miou": miou}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        avg_miou = torch.stack([x["val_miou"] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("val_miou", avg_miou*100., on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()
        

    