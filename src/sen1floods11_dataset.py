import os
from skimage.io import imread
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from pytorch_lightning import LightningDataModule
import albumentations as A

class Sen1Floods11Dataset(Dataset):
    def __init__(self, DF_PATH, split='train', label_type='HandLabeled', debug=False, batch_size=8, transforms=False, processor=None):
        self.label_type = label_type
        self.dataset = pd.read_csv(DF_PATH)
        self.dataset = self.dataset[self.dataset['Split']==split]
        self.batch_size=batch_size
        self.ROOT = os.path.dirname(DF_PATH)
        
        if len(self.dataset) < self.batch_size:
            self.batch_size = len(self.dataset)
        
        if transforms:
            # define augmentation transforms
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(270),
                    # A.ElasticTransform(
                    #     p=0.4, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                    # ),
                    # A.GridDistortion(p=0.4),
                    # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.4),
                ], additional_targets={'image0':'image', 'mask0':'mask'}
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

        row = self.dataset.iloc[index]
        if self.label_type == 'HandLabeled':
            IMG_FOLDER = 'S1Hand'
            LABEL_FOLDER = 'LabelHand'
            WATER_FOLDER = 'JRCWaterHand'
            OTSU_FOLDER = 'S1OtsuLabelHand'
            SPLIT_ROOT = os.path.join(self.ROOT, 'flood_events', 'HandLabeled')
            
            water_path = os.path.join(SPLIT_ROOT, WATER_FOLDER, f'{row["Region"]}_{row["Img_Id"]}_{WATER_FOLDER}.tif')
            otsu_path = os.path.join(SPLIT_ROOT, OTSU_FOLDER, f'{row["Region"]}_{row["Img_Id"]}_{OTSU_FOLDER}.tif')

        elif self.label_type == 'WeaklyLabeled':
            IMG_FOLDER = 'S1Weak'
            LABEL_FOLDER = 'S2IndexLabelWeak'
            SPLIT_ROOT = os.path.join(self.ROOT, 'flood_events', 'WeaklyLabeled')
            
            water_path = None
            otsu_path = None
            
        img_path = os.path.join(SPLIT_ROOT, IMG_FOLDER, f'{row["Region"]}_{row["Img_Id"]}_{IMG_FOLDER}.tif')
        label_path = os.path.join(SPLIT_ROOT, LABEL_FOLDER, f'{row["Region"]}_{row["Img_Id"]}_{LABEL_FOLDER}.tif')
           
        # load vv and vh images
        img = imread(img_path)
        label = imread(label_path)
        water = imread(water_path) if water_path else None
        otsu = imread(otsu_path) if otsu_path else None

        # Calculate the minimum and maximum values for each channel
        min_values = np.min(img, axis=(1, 2))  # Minimum values for each channel
        max_values = np.max(img, axis=(1, 2))  # Maximum values for each channel

        # Min-max normalization for each channel
        img = (img - min_values[:, np.newaxis, np.newaxis]) / (max_values - min_values)[:, np.newaxis, np.newaxis]
        # Convert NaNs to -1
        img = np.nan_to_num(img, -1)
        
        # apply augmentations if specified
        if self.transform:
            # Transpose the image and mask from (C, H, W) to (H, W, C)
            img = np.transpose(img, (1, 2, 0))
            # label = np.transpose(label, (1, 2, 0))

            # Apply the transformations
            augmented = self.transform(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask']

            # Transpose the image and mask back to (C, H, W)
            img = np.transpose(img, (2, 0, 1))
            # label = np.transpose(label, (2, 0, 1))

        # if self.processor:
        #     example["img"] = self.processor(images=img, return_tensors='pt')['pixel_values'].squeeze()
        # else:
        #     example["img"] = torch.from_numpy(rgb_image).permute(2,0,1).float()
        # example["mask"] = torch.from_numpy(label).float()
        # example['water'] = water
        
        example['img'] = img
        example['label'] = label
        if self.label_type == 'HandLabeled':
            example['water'] = water
            example['otsu'] = otsu

        return example
    
    def get_classes(self):
        class_counts = self.dataset['Region'].value_counts()
        return [1/class_counts[i] for i in self.dataset.Region.values]
    
class Sen1Floods11DataModule(LightningDataModule):
    def __init__(self, path, label_type='HandLabeled', batch_size=8, num_workers=0, debug=False, transforms=False, processor=None, **kwargs):
        super().__init__(**kwargs)
        ROOT = os.getcwd()
        self.path = path
        self.hand_labeled_path = os.path.join(os.path.dirname(self.path), 'hand_labeled.csv')
        self.label_type = label_type
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.debug=debug
        self.transforms=transforms
        self.processor=processor
        
    def prepare_data(self):
        self.train_dataset=Sen1Floods11Dataset(self.path, 'train', self.label_type, self.debug, self.batch_size, self.transforms, self.processor)
        self.val_dataset=Sen1Floods11Dataset(self.path, 'valid', self.label_type, self.debug, self.batch_size, self.processor)
        self.test_dataset=Sen1Floods11Dataset(self.hand_labeled_path, 'test', 'HandLabeled', self.debug, self.batch_size, self.processor)
        self.holdout_dataset=Sen1Floods11Dataset(self.hand_labeled_path, 'hold out', 'HandLabeled', self.debug, self.batch_size, self.processor)
        
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
        data_test = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        
        data_holdout = DataLoader(
            self.holdout_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
    
        return [data_test, data_holdout]
    
    def holdout_dataloader(self):
        return DataLoader(
            self.holdout_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )