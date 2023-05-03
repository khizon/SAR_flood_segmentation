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

def segTransformer(image, mask):
    img_w, img_h, _ = image.shape
    
    image = TF.to_pil_image(image.astype("uint8"))
    mask = TF.to_pil_image(mask.astype("uint8"))
    #Random horizontal flipping:
    if np.random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
        
    #Random rotate:
    if np.random.random() > 0.5:
        angle = np.random.uniform(-30, 30)
        image = TF.rotate(image, angle, fill=(255,255,255))
        mask = TF.rotate(mask, angle, fill=(0,))
        
    #Random Affine
    if np.random.random() > 0.4:
        affine_param = T.RandomAffine.get_params(
            degrees = [-180, 180], translate = [0.3,0.3],  
            img_size = [img_w, img_h], scale_ranges = [1, 1.3], 
            shears = [2,2])
        image = TF.affine(image, 
                          affine_param[0], affine_param[1],
                          affine_param[2], affine_param[3], fill=(255,255,255)
                         )
        mask = TF.affine(mask, 
                         affine_param[0], affine_param[1],
                         affine_param[2], affine_param[3], fill=(0,)
                        )
    
    image = np.array(image)
    mask = np.array(mask)
    
    
    return {
        'image': image,
        'mask': mask
    }

class ETCIDataset(Dataset):
    def __init__(self, dataframe, split, debug=False, batch_size=8, transforms=False):
        self.split = split
        self.dataset = pd.read_csv(dataframe)
        self.batch_size=batch_size
        if transforms:
            self.transform = segTransformer
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

        # apply augmentations if specified
        if self.transform:
            augmented = self.transform(rgb_image, flood_mask)
            rgb_image = augmented['image']
            flood_mask = augmented['mask']

        example["image"] = rgb_image.transpose((2, 0, 1)).astype("float32")
        example["mask"] = flood_mask.astype("float32")
        example['water'] = water.astype("float32")

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