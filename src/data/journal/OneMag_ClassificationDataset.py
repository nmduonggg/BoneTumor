import numpy as np
import pandas as pd
import os
import pickle
import random
import cv2
import torch
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

from functools import lru_cache

from PIL import Image

from data import utils
import albumentations as A

# def apply_threshold_mapping(image, target_colors, tolerance):
    
#     masks = []
#     for idx, color in enumerate(target_colors):
#         color = np.array(color)
#         mask = np.all(np.abs(image - color) < tolerance, axis=-1)
#         # output[mask] = color
#         # output[mask] = idx
#         masks.append(mask.mean())
#     return np.argmax(np.array(masks))

def apply_threshold_mapping(image, target_colors, tolerance):
    """
    Maps each pixel in the image to the index of the closest target color within a given tolerance.
    
    Parameters:
    - image: np.ndarray of shape (H, W, 3) representing the RGB image.
    - target_colors: list or np.ndarray of shape (N, 3) representing target colors.
    - tolerance: int or float, threshold for color matching.

    Returns:
    - int: Index of the target color that has the highest average match.
    """
    image = np.asarray(image)  # Ensure input is a NumPy array
    target_colors = np.asarray(target_colors)  # Convert list to NumPy array
    target_colors = target_colors.reshape(1, 1, -1, 3)  # Reshape for broadcasting

    # Compute absolute difference and check if within tolerance
    diff = np.abs(image[..., None, :] - target_colors)  # (H, W, N, 3)
    mask = np.all(diff < tolerance, axis=-1)  # (H, W, N), True if all channels match

    # Compute the mean match rate for each target color
    match_rates = mask.mean(axis=(0, 1))  # (N,)

    # Return the index of the target color with the highest match rate
    return np.argmax(match_rates)



class OneMag_ClassificationDataset(Dataset):
    def __init__(self, opt):
        
        self.image_dir = opt['image_dir']
        self.label_dir = opt['label_dir']
        self.cases = opt['cases']
        
        metadatas = list()
        
        self.case_path_mapping = {}
        
        for case in self.cases:
            patch_case_path = os.path.join(self.image_dir, f"Case{case}", "patches")
            label_case_path = os.path.join(self.label_dir, f"Case_{case}", "patches")
            
            self.case_path_mapping[f"Case{case}"] = {
                'patch': patch_case_path,
                'label': label_case_path
            }
            metadata_file = os.path.join(self.image_dir, f"Case{case}", "metadata.json")
            
            with open(metadata_file, 'r') as f:
                metadatas += json.load(f)
                
        self.metadatas = metadatas
        
        # self.size = (opt['height'], opt['width']) if opt['height'] is not None else None
        self.opt = opt
        self.crop_sz = 256
        self.n_classes = 7
            
        self.target_colors = [
            [255, 255, 255],    # background
            [0, 128, 0],    # Viable tumor
            [255, 143, 204],    # Necrosis
            [255, 0, 0],    # Fibrosis/Hyalination
            [0, 0, 0],  # Hemorrhage/ Cystic change
            [165, 42, 42],  # Inflammatory
            [0, 0, 255]]    # Non-tumor tissue
        self.tolerance = 50
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        
        self.augmentation = A.Compose([
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            ]
        )
        self.cropper = A.RandomCrop(width=self.crop_sz, height=self.crop_sz)
        self.eval_cropper = A.CenterCrop(width=self.crop_sz, height=self.crop_sz)
        
    def random_crop(self, image, mask, random=True):
        if random:
            cropped = self.cropper(image=image, mask=mask)
        else:
            cropped = self.eval_cropper(image=image, mask=mask)
        image = cropped['image']
        mask = cropped['mask']

        return image, mask

    # @lru_cache(maxsize=1024)  # Cache up to 1024 images
    def load_image(self, path):
        return cv2.imread(path)

        
    def __len__(self):
        return len(self.metadatas)
    
    def __getitem__(self, index):
        
        data = self.metadatas[index]
        incase_id = data['incase_id']
        data_case = data['case']
        patch_folder = self.case_path_mapping[data_case]['patch']
        label_folder = self.case_path_mapping[data_case]['label']
        
        x = self.load_image(os.path.join(patch_folder, f"incase_{incase_id}.jpg"))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        y = self.load_image(os.path.join(label_folder, f"incase_{incase_id}.png"))[:, :, :3]
        y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
        
        if self.opt['random']:
            x, y = self.random_crop(image=x, mask=y, random=True)
        else: 
            x, y = self.random_crop(image=x, mask=y, random=False)
        
        if self.opt['augment']:
            augmented = self.augmentation(image=x, mask=y)
            x = augmented['image']
            y = augmented['mask']
        
        y_ = apply_threshold_mapping(y, self.target_colors, self.tolerance)
        x = cv2.resize(x, (128, 128))
        
        x_ = self.transform(x).float()
        # y_ = torch.from_numpy(y_).long()
        
        return x_, y_
