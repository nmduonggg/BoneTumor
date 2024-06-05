import torch  
import torch.nn as nn  
import torch.nn.functional as F

import timm

class GigaPath(nn.Module):
    def __init__(self, out_nc):
        super(GigaPath, self).__init__()
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, img_size=1024)
        self.tile_encoder = model
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512), nn.ReLU(),
            nn.Linear(512, out_nc)
        )
        
    def forward(self, x):
        with torch.no_grad():
            feature = self.tile_encoder(x)
        out = self.classifier(feature)
        return out