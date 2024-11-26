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

class ClassificationDataset_MultiMag(Dataset):
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
        
        self.transform = transforms.Compose(
            [
                # transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        
        self.scales = [0, 1, 2]
        # self.crop = A.CenterCrop(width=256, height=256)

        self.augmentation = A.Compose([
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),]
        )
        
    def center_crop(self, image, mask, h, w):
        cropper = A.CenterCrop(width=w, height=h)
        return cropper(image=image, mask=mask)
        
    def random_scale_crop(self, image, mask):
        h, w = image.shape[:2]
        hat = np.random.choice([0, 1, 2, 3])
        scale = 2 ** hat
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
        mask = cv2.resize(mask, (new_w, new_h))
        augmented = self.crop(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
        
        return image, mask, hat
    
    def scale_crop(self, image, mask):
        h, w = image.shape[:2]
        # hat = np.random.choice([0, 1, 2, 3])
        images, masks = list(), list()
        for hat in self.scales:
            scale = 2 ** hat
            new_h, new_w = int(h // scale), int(w // scale)
            augmented = self.center_crop(image=image, mask=mask, h=new_h, w=new_w)
            image, mask = augmented['image'], augmented['mask']
            # image = cv2.resize(image, (new_w, new_h))
            # mask = cv2.resize(mask, (new_w, new_h))
            images.append(image)
            masks.append(mask)
        return images, masks, self.scales
        
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
            # for i, x in enumerate(xs):
            #     xs[i] = self.augmentation(image=x)['image']
            x = self.augmentation(image=x)['image']
        
        # x, y, hat = self.random_scale_crop(x, y) # randm magnification
        xs, ys, _ = self.scale_crop(x, y)
        for i, y in enumerate(ys):
            ys[i] = apply_threshold_mapping(y, self.target_colors, self.tolerance)
            ys[i] = torch.tensor(ys[i]).long()
        
        # y = apply_threshold_mapping(y, self.target_colors, self.tolerance)
        
        # if abs(np.mean(x) - 255) < 20:
        #     y = 0
        
        for i, x in enumerate(xs):
            xs[i] = self.transform(x).float()
        # y = torch.tensor(y).long() 
        
        # x = torch.stack(xs, dim=0)  # NxCxHxW
        y = ys[-1]
        
        x2, x1, x0 = xs
        
        return x2, x1, x0, y, self.scales[-1]
