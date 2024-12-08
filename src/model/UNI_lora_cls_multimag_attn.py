import os

import torch  
from torch import nn, einsum
import torch.nn.functional as F

import timm
import loralib as lora
from huggingface_hub import hf_hub_download

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


### modules
    
class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        
        self.pre_norm = nn.LayerNorm(dim)
        
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        x_qkv = self.pre_norm(x_qkv)
        
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
    

class UNI_lora_cls_MultiMag_Attn(nn.Module):
    def __init__(self, out_nc):
        super(UNI_lora_cls_MultiMag_Attn, self).__init__()
        
        model = timm.create_model("hf-hub:MahmoodLab/uni", img_size=256,
                                  pretrained=True, init_values=1e-5, dynamic_img_size=True)
        # model = timm.create_model(
        #     "vit_large_patch16_224", img_size=256, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        # )
        self.scales = [0, 1, 2, 3]
        
        self.tile_encoder = model
        
        
        self.attn_01 = CrossAttention(dim=1024, heads=8, dim_head=64, dropout=0.)
        self.attn_02 = CrossAttention(dim=1024, heads=8, dim_head=64, dropout=0.)
        
        self.classifier = nn.Sequential(
            nn.Linear(1024*3, 512), nn.ReLU(),
            nn.Linear(512, out_nc)
        )
        
        self.apply_lora_to_vit(16, 32)

    def encode(self, x):
        # Forward pass through the ViT model with LoRA
        feature = self.tile_encoder(x)
        return feature
        
    def forward(self, x0, x1, x2, scale=None):
        """x0: original version. xn: n-downscaled versions"""
        bs, c, h, w = x0.shape
        feat_0 = self.tile_encoder.forward_features(x0)
        feat_1 = self.tile_encoder.forward_features(x1)
        feat_2 = self.tile_encoder.forward_features(x2)
        
        cls_0, feat_0 = feat_0[:, 0, :], feat_0[:, 1:, :]
        cls_1, feat_1 = feat_1[:, 0, :], feat_1[:, 1:, :]
        cls_2, feat_2 = feat_2[:, 0, :], feat_2[:, 1:, :]
        
        feat_01 = torch.cat([cls_0.unsqueeze(1), feat_1])
        feat_02 = torch.cat([cls_0.unsqueeze(1), feat_2])
        
        attn_01 = self.attn_01(feat_01)[:, 0, :]
        attn_02 = self.attn_02(feat_02)[:, 0, :]
        
        out_feature = torch.cat([cls_0, attn_01, attn_02], dim=-1)
        
        out = self.classifier(out_feature)
        
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