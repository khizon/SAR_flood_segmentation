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

def s1_to_multi(img):
    S = []
    VV = img[0]
    VH = img[1]
    S.append(VV)
    S.append(VH)
    S.append(VV+VH)
    S.append(VH-VV)
    S.append(VV*VV)
    S.append(VH*VH)
    S.append(VV*VH)
    S.append((VV+VH)*(VH-VV))
    multi_image = np.stack(S, axis=0)
    return multi_image

def s1_to_ratios(img):
    S = []
    VV = img[0]
    VH = img[1]
    S.append(VV)
    S.append(VH)
    S.append(np.nan_to_num(VV/VH, -999))
    S.append(np.nan_to_num((VV-VH)/(VV+VH), -999))
    S.append(np.nan_to_num((VH)/(VV+VH), -999))
    S.append(np.nan_to_num((VV)/(VV+VH), -999))
    S.append(np.nan_to_num((4*VH)/(VV+VH), -999))
    
    multi_image = np.stack(S, axis=0)
    return multi_image

class Sen1Floods11Dataset(Dataset):
    def __init__(self, DF_PATH, split='train', label_type='HandLabeled', target="Flood", debug=False, in_channels=3, batch_size=8, transforms=[], processor=None, expand=1, filter_data=False, normalize=False):
        self.label_type = label_type
        self.target = target
        self.dataset = pd.read_csv(DF_PATH)
        self.dataset = self.dataset[(self.dataset['Split']==split)]

        if (filter_data):
            self.dataset = self.dataset[
                (self.dataset['NaN Pixels'] == False)
            ]

        if (split=='train') & (expand > 1):
            print(f'Original Training dataset: {len(self.dataset)}')
            self.dataset = self.dataset.sample(n=int(expand*len(self.dataset)), replace=True)
            print(f'Expanded Training dataset: {len(self.dataset)}')
        
        self.batch_size=batch_size
        self.in_channels = in_channels
        self.ROOT = os.path.dirname(DF_PATH)
        
        if len(self.dataset) < self.batch_size:
            self.batch_size = len(self.dataset)
        
        if len(transforms)>0:
            if debug:
                print(f'{split}: {transforms}')
            all_transforms = []
            if 'shiftscalerotate' in transforms:
                all_transforms.append(A.ShiftScaleRotate(shift_limit=0.5, rotate_limit=270))
            if 'shiftscalerotate_10' in transforms:
                all_transforms.append(A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=270))
            if 'crop' in transforms:
                all_transforms.append(A.RandomCrop(width=256, height=256))
            if 'flip' in transforms:
                all_transforms.append(A.HorizontalFlip(p=0.5))
            if 'rotate' in transforms:
                all_transforms.append(A.Rotate(270))
            if 'lighting' in transforms:
                all_transforms.extend([
                    A.RandomBrightness(limit=0.2, p=0.5),
                    A.RandomContrast(limit=0.2, p=0.5),
                ])
            if 'blur' in transforms:
                all_transforms.append(
                    A.OneOf([
                        A.Blur(blur_limit=5, p=1.0),
                        A.GaussianBlur(blur_limit=5, p=1.0),
                        A.MedianBlur(blur_limit=5, p=1.0),
                        A.MotionBlur(blur_limit=5, p=1.0),
                        A.AdvancedBlur(blur_limit=5, p=1.0),
                    ], p=0.3),
                )
            if 'noise' in transforms:
                all_transforms.append(
                    A.OneOf([
                        A.GaussNoise(var_limit=(10, 50), p=1.0),
                        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                    ], p=0.5),
                )
            if 'perspective' in transforms:
                all_transforms.append(A.Perspective(pad_val=999, mask_pad_val=-1, p=0.25))
            if 'distort' in transforms:
                all_transforms.extend([
                    A.ElasticTransform(p=0.4, alpha=120, sigma=120 * 0.05),
                    A.GridDistortion(p=0.4),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.4),
                ])
            if 'elastic' in transforms:
                all_transforms.append(
                   A.ElasticTransform(
                            p=0.4, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                        ) 
                )
            if 'griddistort' in transforms:
                all_transforms.append(
                    A.GridDistortion(p=0.4)
                )
            if 'coarse_dropout' in transforms:
                all_transforms.append( 
                    A.CoarseDropout(max_holes=8, max_height=4, max_width=4, min_holes=1, min_height=1, min_width=1, fill_value=999, mask_fill_value=-1, p=0.4)
                )
            # define augmentation transforms
            self.transform = A.Compose(all_transforms, additional_targets={'image0':'image', 'mask0':'mask'}
            )
        else:
            self.transform = None
        
        if debug:
            # Return only 2 batch worth of data
            self.dataset = self.dataset.sample(2*self.batch_size, replace=True)

        self.processor = processor

        self.normalize = normalize

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

        if self.in_channels==7:
            # Make the image multi channel
            img = s1_to_ratios(img)
        elif self.in_channels==8:
            img = s1_to_multi(img)  
        else:
            # Create a 3rd channel using ratio of VV and VH layers
            img = s1_to_rgb(img)

        if self.normalize:
            img = self.clip_and_normalize(img)

        if self.target == 'Water':
            label = water
        elif self.target == 'Combined':
            label = np.where(water == 1, 1, label)
            
        # Convert unlabeled pixels to 0
        # label = np.where(label == -1, 0, label)
            
        # Convert NaNs to 999
        if not self.normalize:
            img = np.nan_to_num(img, 999)
        label = np.nan_to_num(label, -1)
        
        # apply augmentations if specified
        if self.transform:
            # Transpose the image and mask from (C, H, W) to (H, W, C)
            img = np.transpose(img, (1, 2, 0))

            # Apply the transformations
            augmented = self.transform(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask']

            # Transpose the image and mask back to (C, H, W)
            img = np.transpose(img, (2, 0, 1))

        
        example['img'] = img
        example['label'] = label
        if self.label_type == 'HandLabeled':
            example['water'] = water
            example['otsu'] = otsu

        return example
    
    def get_classes(self):
        class_counts = self.dataset['Region'].value_counts()
        return [1/class_counts[i] for i in self.dataset.Region.values]

    def clip_and_normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Clip and normalize a 3-channel dB image to [0, 255].

        Parameters
        ----------
        img : np.ndarray
            Input image of shape (H, W, 3) or (3, H, W), with pixel values in dB.

        Returns
        -------
        np.ndarray
            Normalized image of shape (3, H, W), dtype=np.float32, values in [0, 255].
        """
        # Ensure channel-last format for processing
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # (H, W, 3)

        # Clip to [-30, 0] dB
        img_clipped = np.clip(img, -30, 0)

        # Normalize to [0, 255]
        img_norm = (img_clipped - (-30)) / (0 - (-30)) * 255.0

        # Convert back to channel-first (3, H, W)
        img_cf = np.transpose(img_norm, (2, 0, 1))

        return img_cf.astype(np.float32)

class Sen1Floods11DataModule(LightningDataModule):
    def __init__(self, path, label_type='HandLabeled', target='Flood', batch_size=8, num_workers=0, debug=False, transforms=False, in_channels=3, processor=None, expand=1, filter_data=False, normalize=False, **kwargs):
        super().__init__(**kwargs)
        ROOT = os.getcwd()
        self.path = path
        self.hand_labeled_path = os.path.join(os.path.dirname(self.path), 'hand_labeled.csv')
        self.label_type = label_type
        self.target = target
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.debug=debug
        self.transforms=transforms
        self.in_channels = in_channels
        self.processor=processor
        self.expand=expand
        self.filter_data=filter_data
        self.normalize=normalize
        
    def prepare_data(self):
        self.train_dataset=Sen1Floods11Dataset(self.path, 'train', self.label_type, self.target, self.debug, self.in_channels, self.batch_size, self.transforms, self.processor, self.expand, self.filter_data, self.normalize)
        self.val_dataset=Sen1Floods11Dataset(self.path, 'valid', self.label_type, self.target, self.debug, self.in_channels, self.batch_size, [], self.processor, 1 , self.filter_data, self.normalize)
        self.test_dataset=Sen1Floods11Dataset(self.hand_labeled_path, 'test', 'HandLabeled', self.target, self.debug, self.in_channels, self.batch_size, [], self.processor, 1, self.filter_data, self.normalize)
        self.holdout_dataset=Sen1Floods11Dataset(self.hand_labeled_path, 'hold out', 'HandLabeled', self.target, self.debug, self.in_channels, self.batch_size, [], self.processor, 1, self.filter_data, self.normalize)
        
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