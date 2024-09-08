import numpy as np
import pandas as pd
import os
import pickle
import random
import cv2
import torch
import json
from torch.utils.data import Dataset

from data import utils

def apply_threshold_mapping(image, target_colors, tolerance):
    # Create masks for pixels that are closer to green or pink
    # Initialize the output image with the original image
    output = np.zeros_like(image)[:, :, 0] # 2D only
    masks = []
    for idx, color in enumerate(target_colors):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        # output[mask] = color
        output[mask] = idx

    return output

class SegmentDataset(Dataset):
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
            [255, 255, 255],
            [0, 128, 0],
            [255, 143, 204],
            [255, 0, 0],
            [0, 0, 0],
            [165, 42, 42],
            [0, 0, 255]]
        self.tolerance = 50
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        
        item_idx = self.indices[index]
        x = cv2.imread(os.path.join(self.image_dir, f"patch_{item_idx}.png"))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        y = cv2.imread(os.path.join(self.label_dir, f"gt_{item_idx}.png"))
        y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
        y = apply_threshold_mapping(y, self.target_colors, self.tolerance)
        
        
        x = x / 255.0
        # x = utils.normalize_np(x)
        x = torch.tensor(x).permute(2,0,1)
        # x = utils.imresize(x.unsqueeze(0), self.size).squeeze(0)
        
        x = x.float()
        y = torch.tensor(y).long() 
        
        return x, y
