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

from PIL import Image

from data import utils
import albumentations as A

def apply_threshold_mapping(image, target_colors, tolerance):
    
    masks = []
    for idx, color in enumerate(target_colors):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        # output[mask] = color
        # output[mask] = idx
        masks.append(mask.mean())
    return np.argmax(np.array(masks))

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
                transforms.Resize(224),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        

        self.augmentation = A.Compose([
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            ]
        )
        
    def random_crop(self, image, mask):
        
        cropper = A.RandomCrop(width=self.crop_sz, height=self.crop_sz)
        cropped = cropper(image=image, mask=mask)
        image = cropped['image']
        mask = cropped['mask']

        return image, mask
        
    def __len__(self):
        return len(self.metadatas)
    
    def __getitem__(self, index):
        
        data = self.metadatas[index]
        incase_id = data['incase_id']
        data_case = data['case']
        patch_folder = self.case_path_mapping[data_case]['patch']
        label_folder = self.case_path_mapping[data_case]['label']
        
        x = np.array(
            Image.open(os.path.join(patch_folder, f"incase_{incase_id}.jpg")).convert("RGB"))
        y = cv2.imread(os.path.join(label_folder, f"incase_{incase_id}.png"))[:, :, :3]
        y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
        
        
        if self.opt['augment']:
            augmented = self.augmentation(image=x, mask=y)
            x = augmented['image']
            y = augmented['mask']
        
        x_, y_ = self.random_crop(image=x, mask=y)
        y_ = apply_threshold_mapping(y_, self.target_colors, self.tolerance)
        
        x_ = self.transform(x_).float()
        y_ = torch.tensor(y_).long() 
        
        return x_, y_
