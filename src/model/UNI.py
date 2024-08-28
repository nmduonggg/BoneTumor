import os

import torch  
import torch.nn as nn  
import torch.nn.functional as F

import timm
from huggingface_hub import hf_hub_download

class UNI(nn.Module):
    def __init__(self, out_nc):
        super(UNI, self).__init__()
        local_dir = "/home/admin/duongnguyen/BoneTumor/src/weights"
        os.makedirs(local_dir, exist_ok=True)
        # hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        self.tile_encoder = model
        # transform = create_transform(**resolve_data_config(self.tile_encoder.pretrained_cfg, model=self.tile_encoder))
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512), nn.ReLU(),
            nn.Linear(512, out_nc)
        )
        
    def encode(self, x):
        with torch.no_grad():
            feature = self.tile_encoder(x)
        return feature
        
    def forward(self, x):
        with torch.no_grad():
            feature = self.tile_encoder(x)
        out = self.classifier(feature)
        return out