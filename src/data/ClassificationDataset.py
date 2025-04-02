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

class ClassificationDataset(Dataset):
    def __init__(self, opt):
        
        self.image_dir = opt['image_dir']
        self.label_dir = opt['label_dir']
        # self.size = (opt['height'], opt['width']) if opt['height'] is not None else None
        self.opt = opt
        self.n_classes = 7
        
        print("Number of classes: ", self.n_classes)
        with open(opt['label_map'], 'r') as f:
            self.indices = json.load(f)[opt['type']]
            
        self.target_colors = [
            [255, 255, 255],    # background
            [0, 128, 0],    # Viable tumor
            [255, 143, 204],    # Necrosis
            [255, 0, 0],    # Fibrosis/Hyalination
            [0, 0, 0],  # Hemorrhage/ Cystic change
            [165, 42, 42],  # Inflammatory
            [0, 0, 255]]    # Non-tumor tissue
        self.tolerance = 50
        
        self.scales = [0, 1, 2]
        self.transform = transforms.Compose(
            [
                # transforms.Resize(224),
                transforms.ToTensor(),
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
        
    def center_crop(self, image, mask, h, w, scales):
        
        hat = scales[-1]
        scale = 2 ** hat
        new_h, new_w = int(h // scale), int(w // scale)
        cropper = A.RandomCrop(width=new_w, height=new_h)
        cropped = cropper(image=image, mask=mask)
        image = cv2.resize(cropped['image'], (w, h))
        mask = cv2.resize(cropped['mask'], (w, h), 
                            interpolation=cv2.INTER_NEAREST)

        return image, mask
    
    def scale_crop(self, image, mask):
        h, w = image.shape[:2]
        # hat = np.random.choice([0, 1, 2, 3])
        image, mask = self.center_crop(image=image, mask=mask,
                                         h=h, w=w, scales=self.scales)
            
        return image, mask
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        
        item_idx = self.indices[index]
        # x = cv2.imread(os.path.join(self.image_dir, f"patch_{item_idx}.png"))
        # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = np.array(
            Image.open(os.path.join(self.image_dir, f"patch_{item_idx}.png")).convert("RGB"))
        
        y = cv2.imread(os.path.join(self.label_dir, f"gt_{item_idx}.png"))[:, :, :3]
        y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
        if self.opt['augment']:
            augmented = self.augmentation(image=x, mask=y)
            x = augmented['image']
            y = augmented['mask']
        
        x_, y_ = self.scale_crop(image=x, mask=y)
        y_ = apply_threshold_mapping(y_, self.target_colors, self.tolerance)
        
        # if self.opt['augment']:
        #     x_ = self.augmentation(image=x_)['image']
        
        x_ = self.transform(x_).float()
        y_ = torch.tensor(y_).long() 
        
        return x_, y_
