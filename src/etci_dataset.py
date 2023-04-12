"""
Referenced from:
https://medium.com/cloud-to-street/jumpstart-your-machine-learning-satellite-competition-submission-2443b40d0a5a
"""

# import cv2
from skimage.io import imread
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


def s1_to_rgb(vv_image, vh_image):
    ratio_image = np.clip(np.nan_to_num(vh_image / vv_image, 0), 0, 1)
    rgb_image = np.stack((vv_image, vh_image, 1 - ratio_image), axis=2)
    return rgb_image


class ETCIDataset(Dataset):
    def __init__(self, dataframe, split, debug=False, transform=None):
        self.split = split
        self.dataset = pd.read_csv(dataframe)
        self.transform = transform
        
        if debug:
            self.dataset = self.dataset.sample(8)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        example = {}

        df_row = self.dataset.iloc[index]

        # load vv and vh images
        vv_image = imread(df_row["vv"], 0)[:,:,0] / 255.0
        vh_image = imread(df_row["vh"], 0)[:,:,0] / 255.0

        # convert vv and vh images to rgb
        rgb_image = s1_to_rgb(vv_image, vh_image)

        if self.split == "test":
            # no flood mask should be available
            example["image"] = rgb_image.transpose((2, 0, 1)).astype("float32")
        else:
            # load ground truth flood mask
            flood_mask = imread(df_row["flood_label"], 0)[:,:,0] / 255.0
            # flood_mask = np.clip(flood_mask, 0,1)

            # apply augmentations if specified
            if self.transform:
                augmented = self.transform(image=rgb_image, mask=flood_mask)
                rgb_image = augmented["image"]
                flood_mask = augmented["mask"]

            example["image"] = rgb_image.transpose((2, 0, 1)).astype("float32")
            # example["image"] = rgb_image
            example["mask"] = flood_mask.astype("float32")

        return example
    
class ETCIDataModule(LightningDataModule):
    def __init__(self, path, batch_size, num_workers=0, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.debug=debug
        
    def prepare_data(self):
        self.train_dataset=ETCIDataset(self.path+'train.csv', 'train', self.debug)
        self.val_dataset=ETCIDataset(self.path+'val.csv', 'val', self.debug)
        self.test_dataset=ETCIDataset(self.path+'test.csv', 'val', self.debug)
        
    def setup(self, stage=None):
        self.prepare_data()
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )