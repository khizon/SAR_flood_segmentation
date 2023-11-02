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
import albumentations as A


def s1_to_rgb(vv_image, vh_image):
    ratio_image = np.clip(np.nan_to_num(vh_image / vv_image, 0), 0, 1)
    rgb_image = np.stack((vv_image, vh_image, 1 - ratio_image), axis=2)
    return rgb_image

class ETCIDataset(Dataset):
    def __init__(self, dataframe, split, debug=False, batch_size=8, transforms=False, processor=None):
        self.split = split
        self.dataset = pd.read_csv(dataframe)
        self.dataset = self.dataset[
            (self.dataset['invalid']!=True)
           ]
        self.batch_size=batch_size
        if transforms:
            # define augmentation transforms
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(270),
                    A.ElasticTransform(
                        p=0.4, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                    ),
                    A.GridDistortion(p=0.4),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.4),
                ]
            )
        else:
            self.transform = None
        
        if debug:
            # Return only 1 batch worth of data
            self.dataset = self.dataset.sample(self.batch_size)
        self.processor = processor

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

        # apply augmentations if specified
        if self.transform:
            augmented = self.transform(image=rgb_image, mask=flood_mask)
            rgb_image = augmented['image']
            flood_mask = augmented['mask']

        if self.processor:
            example["image"] = self.processor(images=rgb_image, return_tensors='pt')['pixel_values'].squeeze()
        else:
            example["image"] = torch.from_numpy(rgb_image).permute(2,0,1).float()
        example["mask"] = torch.from_numpy(flood_mask).float()
        example['water'] = water

        return example
    
    def get_classes(self):
        class_counts = self.dataset['has_mask'].value_counts()
        return [1/class_counts[i] for i in self.dataset.has_mask.values]
        
    
class ETCIDataModule(LightningDataModule):
    def __init__(self, path, batch_size, num_workers=0, debug=False, transforms=False, processor=None, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.debug=debug
        self.transforms=transforms
        self.processor=processor
        
    def prepare_data(self):
        self.train_dataset=ETCIDataset(self.path+'train.csv', 'train', self.debug, self.batch_size, self.transforms, self.processor)
        self.val_dataset=ETCIDataset(self.path+'val.csv', 'val', self.debug, self.batch_size, self.processor)
        self.test_dataset=ETCIDataset(self.path+'test.csv', 'val', self.debug, self.batch_size, self.processor)
        
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
            shuffle=False,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )