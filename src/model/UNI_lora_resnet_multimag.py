import os

import torch  
import torch.nn as nn  
import torch.nn.functional as F

import timm
import loralib as lora
from huggingface_hub import hf_hub_download

class UNI_lora_resnet_MultiMag(nn.Module):
    def __init__(self, out_nc):
        super(UNI_lora_resnet_MultiMag, self).__init__()
        
        model = timm.create_model("hf-hub:MahmoodLab/uni", img_size=256,
                                  pretrained=True, init_values=1e-5, dynamic_img_size=True)
        # model = timm.create_model(
        #     "vit_large_patch16_224", img_size=256, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        # )
        self.scales = [0, 1, 2]
        
        self.tile_encoder = model
        
        self.context_enc1 = timm.create_model('resnet50.a1_in1k', pretrained=True,
                              features_only=True) 
        
        self.context_enc2 = timm.create_model('resnet50.a1_in1k', pretrained=True,
                              features_only=True) 
        
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, out_nc)
        )
        
        self.apply_lora_to_vit(16, 32)

    def encode(self, x):
        # Forward pass through the ViT model with LoRA
        feature = self.tile_encoder(x)
        return feature
        
    def forward(self, x0, x1, x2, scale=None):
        """x0: original version. xn: n-downscaled versions"""
        bs, c, h, w, = x0.shape
        
        # multi-mag context extracting
        
        feat_1 = torch.mean(
            self.context_enc1(x1)[-2].reshape(bs, 1024, -1), dim=-1)
        feat_2 = torch.mean(
            self.context_enc2(x2)[-2].reshape(bs, 1024, -1), dim=-1)
        
        # 20x extracting
        feat_0 = self.tile_encoder(x0)
        
        feature = torch.stack([feat_1, feat_2, feat_0], dim=1)  # bxnx...
        feature = torch.mean(feature, dim=1)    # can be cross attention later
        out = self.classifier(feature)
        
        return out
    
    def full_forward(self, x):
        bs, c, h, w = x.shape
        feature = self.tile_encoder(x)
        out = self.classifier(feature)
        return feature, out
    
    def apply_lora_to_vit(self, lora_r, lora_alpha):
        """
        Apply LoRA to all the Linear layers in the Vision Transformer model.
        """
        # Step 1: Collect the names of layers to replace
        layers_to_replace = []
        
        for name, module in self.tile_encoder.named_modules():
            if isinstance(module, nn.Linear) :
                if 'qkv' in name or 'proj' in name:
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
        # for classifier in self.classifiers:
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        for param in self.context_enc1.parameters():
            param.requires_grad = True
        for param in self.context_enc2.parameters():
            param.requires_grad = True