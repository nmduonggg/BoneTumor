import os
import sys
sys.path.append('../BBDM2')

import torch  
import torch.nn as nn  
import torch.nn.functional as F

import timm
import loralib as lora
from huggingface_hub import hf_hub_download
import numpy as np
import data.utils as data_utils

from model import TransformerReorder, UNI_lora_cls
import BBDM2.model.BrownianBridge.LatentBrownianBridgeModel_PathologyContext as LBBDM

class StackedDiffusionModel(nn.Module):
    def __init__(self, option):
        super(StackedDiffusionModel, self).__init__()
        
        self.phase1_classifier = UNI_lora_cls(option['out_nc'])
        self.phase2_refiner = LBBDM(option.bbdm)
        self.patch_size = 256
        self.phase2_size = 256
        self.color_map = [
            [255, 255, 255],    # background
            [0, 128, 0],    # Viable tumor
            [255, 143, 204],    # Necrosis
            [255, 0, 0],    # Fibrosis/Hyalination
            [0, 0, 0],  # Hemorrhage/ Cystic change
            [165, 42, 42],  # Inflammatory
            [0, 0, 255]    # Non-tumor tissue
        ]
        
    def forward(self, x):
        # BxCxHxW
        batch_size = x.size(0)
        
        # run phase 1
        patch_seq, num_h, num_w, h, w = self._generate_patch_seq(x) # -> BxSx1025
        
        out1 = self._combine_tensor(
                        patch_seq, num_h, num_w, h, w, self.patch_size, self.patch_size,
                        batch_size=batch_size, channel=3)
        
        out1 = data_utils.denormalize_tensor(out1)
        x = data_utils.denormalize_tensor(x)
        
        x_cond = F.interpolate(out1, (self.phase2_size, self.phase2_size))
        x_cont = F.interpolate(x, (self.phase2_size, self.phase2_size))
        
        out = self.phase2_refiner.sample_infer(x_cond, x_cont, clip_denoised=self.option.bbdm.clip_denoised)
        
        return out
    
    def _generate_patch_seq(self, x):
        img_list, num_h, num_w, h, w = self._crop_tensor(x, self.patch_size, self.patch_size)
        
        outs = []
        for img in img_list:
            pred = self.phase1_classifier(img)    # BxN
            out_idx = torch.argmax(pred, dim=-1) # B
            out = torch.zeros_like(pred)
            out.scatter_(1, out_idx.unsqueeze(1), 1)    # BxN
            out = out.unsqueeze(-1) * torch.tensor(self.color_map).unsqueeze(0)    # BxNx1 * 1xNx3 -> BxNx3
            out = torch.sum(out, dim=1) # BxNx3 -> Bx3
            outs.append(out)
        
        return outs, num_h, num_w, h, w
    
    def _crop_tensor(self, img, crop_sz, step):
        # img: BxCxHxW
        b, c, h, w = img.shape
        h_space = np.arange(0, h - crop_sz + 1, step)
        w_space = np.arange(0, w - crop_sz + 1, step)
        index = 0
        num_h = 0
        lr_list=[]
        for x in h_space:
            num_h += 1
            num_w = 0
            for y in w_space:
                num_w += 1
                index += 1
                crop_img = img[:, :, x:x + crop_sz, y:y + crop_sz]
                lr_list.append(crop_img)
        h=x + crop_sz
        w=y + crop_sz
        return lr_list, num_h, num_w, h, w
    
    def _combine_tensor(self, sr_list, num_h, num_w, h, w, patch_size, step, batch_size, channel=3):
        index=0
        device = sr_list[0].device
        
        sr_img = torch.zeros((batch_size, h, w, channel)).to(device)
        for i in range(num_h):
            for j in range(num_w):
                sr_subim = sr_list[index]
                
                sr_img[:, i*step: i*step+patch_size, j*step: j*step+patch_size,:] += \
                    torch.ones((batch_size, step, step, channel)).to(sr_subim.device) * sr_subim
                index+=1

        for j in range(1,num_w):
            sr_img[:,j*step:j*step+(patch_size-step),:]/=2

        for i in range(1,num_h):
            sr_img[i*step:i*step+(patch_size-step),:,:]/=2
        return sr_img
        
    def load_state_dict(self, phase1_dict, phase2_dict, strict=True):
        self.phase1_classifier.load_state_dict(phase1_dict, strict=strict)
        self.phase2_refiner.load_state_dict(phase2_dict['model'], strict=strict)