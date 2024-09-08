import os

import torch  
import torch.nn as nn  
import torch.nn.functional as F

import timm
import loralib as lora
from huggingface_hub import hf_hub_download
from peft import get_peft_model, LoraConfig, TaskType

class UNI(nn.Module):
    def __init__(self, out_nc):
        super(UNI, self).__init__()
        
        model = timm.create_model("hf-hub:MahmoodLab/uni", img_size=1024,
                                  pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.tile_encoder = model
        
        self.segment_head = nn.Sequential(
            nn.Conv2d(4, 4*4*4, 3, 1, 1), nn.PixelShuffle(4), nn.ReLU(),
            nn.Conv2d(4, 4*4*4, 3, 1, 1), nn.PixelShuffle(4), nn.ReLU(),
            nn.Conv2d(4, 4*4*4, 3, 1, 1), nn.PixelShuffle(4), nn.ReLU(),
            nn.Conv2d(4, out_nc, 3, 1, 1)
        )
        # self.apply_lora_to_vit(1, 2)

    def encode(self, x):
        # Forward pass through the ViT model with LoRA
        feature = self.tile_encoder(x)
        return feature
        
    def forward(self, x):
        bs, c, h, w = x.shape
        feature = self.tile_encoder(x)
        feature = feature.reshape(bs, -1, 16, 16)
        out = self.segment_head(feature)
        return out
    
    def apply_lora_to_vit(self, lora_r, lora_alpha):
        """
        Apply LoRA to all the Linear layers in the Vision Transformer model.
        """
        # Step 1: Collect the names of layers to replace
        layers_to_replace = []
        
        for name, module in self.tile_encoder.named_modules():
            if isinstance(module, nn.Linear) :
                if 'qkv' in name:
                    # Collect layers for replacement (store name and module)
                    layers_to_replace.append((name, module))
        
        # Step 2: Replace the layers outside of the iteration
        for name, module in layers_to_replace:
            # Create the LoRA-augmented layer
            lora_layer = lora.Linear(module.in_features, module.out_features, r=lora_r, lora_alpha=lora_alpha)
            # Copy weights and bias
            lora_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_layer.bias.data = module.bias.data.clone()

            # Replace the layer in the model
            parent_name, layer_name = name.rsplit('.', 1)
            parent_module = dict(self.tile_encoder.named_modules())[parent_name]
            setattr(parent_module, layer_name, lora_layer)

    # Additional helper to enable LoRA fine-tuning
    def enable_lora_training(self):
        # Set LoRA layers to be trainable, freeze others
        for param in self.tile_encoder.parameters():
            param.requires_grad = False
        
        for name, param in self.tile_encoder.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        # Enable gradients for the classifier head
        for param in self.segment_head.parameters():
            param.requires_grad = True