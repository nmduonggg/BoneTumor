import os
import sys
sys.path.append('/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/BBDM2')

import torch  
import torch.nn as nn  
import torch.nn.functional as F

import timm
import loralib as lora
from huggingface_hub import hf_hub_download
import numpy as np
import data.utils as data_utils

from model import TransformerReorder, UNI_lora_cls
from model.bbdm.BrownianBridge.LatentBrownianBridgeModel_PathologyContext import LatentBrownianBridgeModel_Pathology as LBBDM

class StackedDiffusionModel(nn.Module):
    def __init__(self, option):
        super(StackedDiffusionModel, self).__init__()
        self.option=option
        self.phase1_classifier = UNI_lora_cls(option['network_G']['out_nc'])
        
        self.phase2_refiner = LBBDM(option['bbdm']['model'])
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
                        batch_size=batch_size, channel=3)   # B, H, W, C
        
        out1 = out1.permute(0, 3, 1, 2) / 255.  # original color 0,1
        
        x = data_utils.denormalize_tensor(x)
        
        x_cond = F.interpolate(out1.to(x.device), (self.phase2_size, self.phase2_size))
        x_cont = F.interpolate(x, (self.phase2_size, self.phase2_size))
        
        x_cond = self._to_normal(x_cond)
        x_cont = self._to_normal(x_cont)
        
        out = self.phase2_refiner.sample_infer(x_cond, x_cont, clip_denoised=self.option['bbdm']['clip_denoised'])
        
        np_out = out.permute(0,2,3,1).squeeze(0).cpu().numpy()
        print(np_out)
        
        return out
    
    def _to_normal(self, x):
        x = (x - 0.5) * 2.
        x = torch.clamp(x, -1, 1)
        return x
    
    def _rm_normal(self, x):
        x = x * 0.5 + 0.5
        return x

    def _generate_patch_seq(self, x):
        img_list, num_h, num_w, h, w = self._crop_tensor(x, self.patch_size, self.patch_size)
        
        outs = []
        img_tensor = torch.stack(img_list, dim=0)   # length_of_seq x B x CxHxW
        length, B, C, H, W = img_tensor.shape
        img_tensor = img_tensor.reshape(length*B, C, H, W).to(x.device)
        
        preds = self.phase1_classifier(img_tensor) 
        preds = preds.reshape(length, B, -1)    # length_of_seq x B x C
        out_indices = torch.argmax(preds, dim=-1)   # SxB
        out = torch.zeros_like(preds)
        out.scatter_(2, out_indices.unsqueeze(2), 1)    # SxBxC
        out = out.unsqueeze(-1) * torch.tensor(self.color_map).reshape(1, 1, len(self.color_map), -1).to(out.device)   # SxBxCx3
        out = torch.sum(out, dim=2)    # SxBx3
        
        # print(out.shape)
        out = out.reshape(list(out.shape) + [1, 1]) * \
            torch.ones(list(out.shape) + [self.patch_size, self.patch_size]).to(out.device)
        # outs = [out[i] for i in range(len(img_list))]
        
        # for i in range(len(img_list)):
        #     # pred = self.phase1_classifier(img)    # BxN
        #     pred = preds[i, ...]    # BxC
        #     out_idx = torch.argmax(pred, dim=-1) # B
        #     out = torch.zeros_like(pred)
        #     out.scatter_(1, out_idx.unsqueeze(1), 1)    # BxN
        #     out = out.unsqueeze(-1) * torch.tensor(self.color_map).to(out.device).unsqueeze(0)    # BxNx1 * 1xNx3 -> BxNx3
        #     out = torch.sum(out, dim=1) # BxNx3 -> Bx3
            
        #     outs.append(out)
        
        return out, num_h, num_w, h, w
    
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
                sr_subim = sr_subim.permute(0,2,3,1)    # BxCxHxW -> BxHxWxC
                
                sr_img[:, i*step: i*step+patch_size, j*step: j*step+patch_size,:] += sr_subim    # BxHxWxC * BxC
                index+=1

        for j in range(1,num_w):
            sr_img[:,j*step:j*step+(patch_size-step),:]/=2

        for i in range(1,num_h):
            sr_img[i*step:i*step+(patch_size-step),:,:]/=2
        return sr_img
        
    def load_state_dict(self, phase1_dict, phase2_dict, strict=True):
        self.phase1_classifier.load_state_dict(phase1_dict, strict=strict)
        self.phase2_refiner.load_state_dict(phase2_dict['model'], strict=strict)