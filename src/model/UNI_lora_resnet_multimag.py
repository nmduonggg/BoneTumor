import os

import torch  
import torch.nn as nn  
import torch.nn.functional as F

import timm
import loralib as lora
from huggingface_hub import hf_hub_download

import model.utils as utils
from torchvision import transforms

class UNI_lora_resnet_MultiMag(nn.Module):
    def __init__(self, out_nc):
        super(UNI_lora_resnet_MultiMag, self).__init__()
        
        # model = timm.create_model("hf-hub:MahmoodLab/uni", img_size=256,
        #                           pretrained=True, init_values=1e-5, dynamic_img_size=True)
        model = timm.create_model(
            "vit_large_patch16_224", img_size=256, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.scales = [0, 1, 2]
        self.out_nc = out_nc
        
        self.tile_encoder = model
        
        self.context_enc1 = timm.create_model('resnet50.a1_in1k', pretrained=True,
                              features_only=True) 
        
        self.context_enc2 = timm.create_model('resnet50.a1_in1k', pretrained=True,
                              features_only=True) 
        
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, out_nc)
        )
                # normalize
        self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
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
        
        # x5 & x10
        feat_0 = torch.mean(
            self.context_enc1(x0)[-2].reshape(bs, 1024, -1), dim=-1)
        feat_1 = torch.mean(
            self.context_enc2(x1)[-2].reshape(bs, 1024, -1), dim=-1)
        
        # 20x extracting
        feat_2 = self.tile_encoder(x2)
        
        feature = torch.stack([feat_1, feat_2, feat_0], dim=1)  # bxnx...
        feature = torch.mean(feature, dim=1)    # can be cross attention later
        out = self.classifier(feature)
        
        return out
    
    def hier_forward(self, im0):
        """
        im0 is original 5x numpy image without preprocessing
        Todo:
            - cut into 10x 20x
            - preprocess each, forward
        """
        device = next(self.classifier.parameters()).device
        
        im0 = self.transform(im0).float().unsqueeze(0)
        x_im0 = torch.cat([im0 for _ in range((256 // 64))], dim=0).to(device)
        im1s, num_h1, num_w1, h1, w1 = utils.crop_tensor(im0, crop_sz=128, step=128)
        
        y_im1s = []
        for im1 in im1s:
            x_im1 = torch.cat([im1 for _ in range(256 // 64)], dim=0).to(device)
            im2s, num_h, num_w, h, w = utils.crop_tensor(im1, crop_sz=64, step=64)
            x_im2 = torch.cat(im2s, dim=0).to(device)
            
            y_im2s = self.forward(x_im0, x_im1, x_im2, None) # BxCl
            # print(y_im2s)
            y_im2s = torch.ones([x_im2.size(0), 1, x_im2.size(2), x_im2.size(3)]).to(device) * y_im2s.reshape(len(im2s), -1, 1, 1)  # BxClxHxW
            
            y_im1 = utils.combine_output(y_im2s, num_h, num_w, h, w, 64, 64, channel=self.out_nc)
            y_im1s.append(y_im1.cpu())
            
        y_im1s = torch.cat(y_im1s, dim=0)
        y_im0 = utils.combine_output(y_im1s, num_h1, num_w1, h1, w1, 128, 128, channel=self.out_nc)
        
        return y_im0
            
            
    
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