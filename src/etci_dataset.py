"""
Referenced from:
https://medium.com/cloud-to-street/jumpstart-your-machine-learning-satellite-competition-submission-2443b40d0a5a
"""

# import cv2
from skimage.io import imread
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from torchvision.transforms import functional as TF
from torchvision import transforms as T


def s1_to_rgb(vv_image, vh_image):
    ratio_image = np.clip(np.nan_to_num(vh_image / vv_image, 0), 0, 1)
    rgb_image = np.stack((vv_image, vh_image, 1 - ratio_image), axis=2)
    return rgb_image


class ETCIDataset(Dataset):
    def __init__(self, dataframe, split, debug=False, batch_size=8, transforms=False):
        self.split = split
        self.dataset = pd.read_csv(dataframe)
        self.dataset = self.dataset[self.dataset['invalid']!=True]
        self.batch_size=batch_size
        if transforms:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(0.5),
                T.RandomApply([
                    T.RandomRotation([-90,90], fill=(1,1,1,0))
                    ], 0.5),
                T.RandomApply([
                    T.ElasticTransform(alpha=120.0, sigma=120 * 0.05, fill=(1,1,1,0))
                    ], 0.4)
            ])
        else:
            self.transform = None
        
        if debug:
            # Return only 1 batch worth of data
            self.dataset = self.dataset.sample(self.batch_size)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        example = {}

        df_row = self.dataset.iloc[index]

        # load vv and vh images
        vv_image = imread(df_row["vv_image_path"], as_gray=True)
        vh_image = imread(df_row["vh_image_path"], as_gray=True)
        flood_mask = imread(df_row["flood_label_path"], as_gray=True)
        water = imread(df_row["water_body_label_path"], as_gray=True)

        # convert vv and vh images to rgb
        rgb_image = s1_to_rgb(vv_image, vh_image)
        
        # convert to tensor
        rgb_image = TF.to_tensor(rgb_image).float()
        flood_mask = TF.to_tensor(flood_mask).float()

        # apply augmentations if specified
        if self.transform:
            to_aug = torch.cat([rgb_image, flood_mask])
            augmented = self.transform(to_aug)
            rgb_image = augmented[:3,:,:]
            flood_mask = augmented[3,:,:,]

        example["image"] = rgb_image
        example["mask"] = flood_mask.squeeze()
        example['water'] = water

        return example
    
    def get_classes(self):
        class_counts = self.dataset['has_mask'].value_counts()
        return [1/class_counts[i] for i in self.dataset.has_mask.values]
        
    
class ETCIDataModule(LightningDataModule):
    def __init__(self, path, batch_size, num_workers=0, debug=False, transforms=False, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.debug=debug
        self.transforms=transforms
        
    def prepare_data(self):
        self.train_dataset=ETCIDataset(self.path+'train.csv', 'train', self.debug, self.batch_size, self.transforms)
        self.val_dataset=ETCIDataset(self.path+'val.csv', 'val', self.debug, self.batch_size)
        self.test_dataset=ETCIDataset(self.path+'test.csv', 'val', self.debug, self.batch_size)
        
    def setup(self, stage=None):
        self.prepare_data()
        
    def train_dataloader(self):
        sampler = WeightedRandomSampler(
            weights=self.train_dataset.get_classes(),
            num_samples=len(self.train_dataset),
            replacement=True
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
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