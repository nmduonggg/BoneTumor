import torch
from torch import nn, einsum
import timm
import loralib as lora

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import model.utils as utils

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

class SimilarityContrastiveLoss(nn.Module):
    """
    Similarity Contrastive Loss:
    - Encourages positive pairs (same index in batch) to have high similarity.
    - Encourages negative pairs (different indices) to have low similarity.
    """

    def __init__(self, margin=0.5):
        """
        :param margin: Margin for negative pairs (default: 0.5)
        """
        super(SimilarityContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2):
        """
        Compute Similarity Contrastive Loss.

        :param x1: Tensor of shape (B, D) - First batch of vectors
        :param x2: Tensor of shape (B, D) - Second batch of vectors
        :return: Similarity contrastive loss
        """
        B, D = x1.shape

        # Compute cosine similarity matrix between all pairs
        cos_sim_matrix = F.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=-1)  # (B, B)

        # Create labels: Positive pairs are diagonal (same index), others are negatives
        labels = torch.eye(B, device=x1.device)  # Identity matrix: 1 on diagonal, 0 elsewhere

        # Loss for positive pairs (diagonal) -> encourage similarity close to 1
        positive_loss = (1 - cos_sim_matrix) * labels  # Only for positive pairs

        # Loss for negative pairs (off-diagonal) -> encourage similarity < margin
        negative_loss = torch.clamp(cos_sim_matrix - self.margin, min=0) * (1 - labels)

        # Compute total loss (average over all pairs)
        loss = positive_loss.sum() + negative_loss.sum()
        loss /= B  # Normalize by batch size

        return loss
    
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


class UNI_lora_multimag_ISBI(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.simCL = SimilarityContrastiveLoss(0.2)
        # login()
        # self.enc1 = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        # self.enc2 = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.enc2 = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        
        self.attn_01 = CrossAttention(dim=1024, heads=8, dim_head=64, dropout=0.1)
        # self.attn_02 = CrossAttention(dim=1024, heads=8, dim_head=64, dropout=0.1)
        
        self.classifier1 = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, num_classes))
        
        self.apply_lora_to_vit(16, 32)
        
        self.transform = transforms.Compose(
            [
                # transforms.Resize(224),
                # transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.out_nc = num_classes
        
    def hier_forward(self, im0):
        """
        im0 is original 5x numpy image without preprocessing
        Todo:
            - cut into 10x 20x
            - preprocess each, forward
        """
        device = next(self.classifier1.parameters()).device
        
        
        ### CROP IMAGES
        ### So we need to padding here in order to ensure that the subimages at corner are processed properly.
        
        print("CKPT 1")
        
        im0 = self.transform(im0).float().unsqueeze(0)
        x_im0 = torch.cat([im0 for _ in range(int((256 // 128)**2))], dim=0).to(device) # /2 for h and w
        x_im0 = x_im0.squeeze(1)
        print("CKPT 2")
        im1s, num_h1, num_w1, h1, w1 = utils.crop_tensor(im0, crop_sz=128, step=128)
        print("CKPT 3")
        x_im1 = torch.cat(im1s, dim=0).to(device)
        print(x_im0.shape, x_im1.shape)
        y_im1s = self.forward(x_im0, x_im1, None)[0] # 4xC
        print("CKPT 4")
            # print(y_im2s)
        
        # print(len(y_im1s), y_im1s[0].shape)
        # y_ims = torch.cat(y_im1s)
            
        y_im1s = torch.ones([x_im1.size(0), 1, x_im1.size(2), x_im1.size(3)]).to(device) * y_im1s.reshape(x_im1.size(0), -1, 1, 1)  # BxClxHxW
        y_im0 = utils.combine_output(y_im1s, num_h1, num_w1, h1, w1, 128, 128, channel=self.out_nc)
        y_im0 = y_im0.permute(1,2,0)
        return y_im0
        
        
    def hier_infer(self, im0):
        """
        im0 is original 5x numpy image without preprocessing
        Todo:
            - cut into 10x 20x
            - preprocess each, forward
        """
        device = next(self.classifier1.parameters()).device
        
        
        ### CROP IMAGES
        ### So we need to padding here in order to ensure that the subimages at corner are processed properly.
        
        print("CKPT 1")
        
        im0 = self.transform(im0).float().unsqueeze(1)
        x_im0 = torch.cat([im0 for _ in range(int((256 // 128)**2))], dim=1).to(device) # /2 for h and w
        x_im0 = x_im0.squeeze(0)
        print("CKPT 2")
        im1s, num_h1, num_w1, h1, w1 = utils.crop_tensor(im0, crop_sz=128, step=128)
        print("CKPT 3")
        x_im1 = torch.stack(im1s, dim=1).to(device)
        print(x_im0.shape, x_im1.shape)
        x_im1 = x_im1.squeeze(0)
        
        if len(x_im0.shape) == 5:
            B, L, C, H, W = im0.shape
            x_im0 = x_im0.reshape(B*L, C, H, W)
            x_im1 = x_im1.reshape(B*L, C, H, W)
        else:
            L, C, H, W = im0.shape
            B = 1
        
        
        y_im1s = self.forward(x_im0, x_im1, None)[0] # 4xC
        print("CKPT 4")
            # print(y_im2s)
        
        # print(len(y_im1s), y_im1s[0].shape)
        # y_ims = torch.cat(y_im1s)
            
        y_im1s = torch.ones([x_im1.size(0), 1, x_im1.size(2), x_im1.size(3)]).to(device) * y_im1s.reshape(x_im1.size(0), -1, 1, 1)  # BxClxHxW
        
        if B>1:        
            y_im1s = y_im1s.reshape(B, L, self.out_nc, H, W)
        y_im0 = utils.combine_batched_output(y_im1s, num_h1, num_w1, h1, w1, B, 128, 128, channel=self.out_nc)
        if B>1:
            y_im0 = y_im0.permute(0,2,3,1)
        else:
            y_im0 = y_im0.permute(1,2,0)
            
        return y_im0
        
        
    def forward(self, x0, x1, scale=None):
        """
        return out of x2
        """
        
        B = x0.size(0)
        # feat = torch.cat([x0, x1])
        
        print("CKPT 1.1")
        feat_0 = self.enc2.forward_features(x0)
        feat_1 = self.enc2.forward_features(x1)
        print("CKPT 1.2.")
        # feat_0 = feat[:B, ...]
        # feat_1 = feat[B:, ...]
        
        cls_0, feat_0 = feat_0[:, 0, :], feat_0[:, 1:, :]
        cls_1, feat_1 = feat_1[:, 0, :], feat_1[:, 1:, :]
        
        fused_all = torch.cat([cls_1.unsqueeze(1), feat_0, feat_1], dim=1)  # use feat 20x attend to 5x and 10x
        # cont_cell = torch.cat([cls_cont.unsqueeze(1), feat_cell], dim=1)
        print("CKPT 1.3")
        fused_cls_2 = self.attn_01(fused_all)[:, 0, :]
        print("CKPT 1.4")
        
        out1 = self.classifier1(
            (cls_1 + fused_cls_2) * 0.5)
        
        sim_loss = self.simCL(fused_cls_2, cls_1)
        
        # the below return is for the best
        return out1, cls_1, sim_loss
    
    def apply_lora_to_vit(self, lora_r, lora_alpha, first_layer_start=15):
        """
        Apply LoRA to all the Linear layers in the Vision Transformer model.
        """
        for enc in [self.enc2]:
            # Step 1: Collect the names of layers to replace
            layers_to_replace = []
            
            for name, module in enc.named_modules():
                if isinstance(module, nn.Linear) :
                    if ('qkv' in name or 'proj' in name) and (int(name.split('.')[1]) >= first_layer_start):
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
                parent_module = dict(enc.named_modules())[parent_name]
                setattr(parent_module, layer_name, lora_layer)

    # Additional helper to enable LoRA fine-tuning
    def enable_lora_training(self):
        # LoRA for enc 1
        # for param in self.enc1.parameters():
        #     param.requires_grad = False
        # for name, param in self.enc1.named_parameters():
        #     if "lora" in name:
        #         param.requires_grad = True
           
        # LoRA for enc 2     
        for param in self.enc2.parameters():
            param.requires_grad = False
        for name, param in self.enc2.named_parameters():
            if "lora" in name:
                param.requires_grad = True
         
        # # LoRA for enc 0       
        # for param in self.enc0.parameters():
        #     param.requires_grad = False
        # for name, param in self.enc0.named_parameters():
        #     if "lora" in name:
        #         param.requires_grad = True

        # Enable gradients for the classifier head
        for param in self.classifier1.parameters():
            param.requires_grad = True
        for param in self.attn_01.parameters():
            param.requires_grad = True

        
    
        