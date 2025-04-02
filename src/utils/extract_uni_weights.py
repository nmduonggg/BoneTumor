import torch
import os

full_weight_path = '..'
out_path = '..'
weight = torch.load(full_weight_path, map_location='cpu')
uni_only_weight = {}
for key, value in weight.items():
    if 'tile_encoder' in key and 'lora' not in key:
        uni_only_weight[key] = value
        
torch.save(uni_only_weight, out_path)
    
