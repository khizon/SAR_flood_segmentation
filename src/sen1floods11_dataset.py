import os
from skimage.io import imread
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from pytorch_lightning import LightningDataModule
import albumentations as A

def s1_to_rgb(img):
    ratio_image = img[0] / img[1]
    rgb_image = np.stack((img[0], img[1], 1-ratio_image), axis=0)
    return rgb_image

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
                    A.RandomCrop(width=256, height=256),
                    # A.Resize(height=512, width=512, interpolation=cv2.INTER_LINEAR, p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(270),
                    A.ElasticTransform(
                        p=0.4, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                    ),
                    A.GridDistortion(p=0.4),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.4),
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

        # Apply mean and std dev normalization using the values reported in the dataset's paper
        data_mean = [0.6851, 0.5235]
        data_sd = [0.0820, 0.1102]
        
        # img = self.normalize_img(img, data_mean, data_sd)
        # Create a 3rd channel using ratio of VV and VH layers
        img = s1_to_rgb(img)
        
        # Convert NaNs to -99
        img = np.nan_to_num(img, 9e5)
        label = np.nan_to_num(label, -1)
        
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
    
    def normalize_img(self, img, mean, std):
        # img shape: C x H x W
        # mean and std are lists of length C
        for c in range(img.shape[0]):
            img[c, :, :] = (img[c, :, :] - mean[c]) / std[c]
        return img
    
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