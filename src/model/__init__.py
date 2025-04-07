from __future__ import absolute_import
# from .mobilenetv2 import mobilenetv2
from .UNI_lora_cls import UNI_lora_cls
from .ViT_baseline import ViT_baseline
from .ResNet_baseline import ResNet_baseline
from .MobileNetV2_baseline import MobileNetV2_baseline
from .ConvNextv2_baseline import ConvNeXtV2_baseline
from .SwinTransformerV2_baseline import SwinTransformerV2_baseline


from .UNet_baseline import UNet
from .SegFormer_baseline import Segformer
from .UNI_lora_cls_multimag import UNI_lora_cls_MultiMag
from .UNI_lora_resnet_multimag import UNI_lora_resnet_MultiMag
from .UNI_lora_cls_multimag_attn import UNI_lora_cls_MultiMag_Attn


from .Transformer_reorder import TransformerReorder
from .Stacked_model import StackedModel
from .StackedDiffusion_model import StackedDiffusionModel
from .DiscreteStackedDiffusion_model import DiscreteStackedDiffusionModel

def create_model(opt):
    if opt['network_G']['which_model_G'] == 'uni_lora_cls':
        return UNI_lora_cls(opt['network_G']['out_nc'])
    if opt['network_G']['which_model_G'] == 'uni_lora_cls_multimag':
        return UNI_lora_cls_MultiMag(opt['network_G']['out_nc'])
    if opt['network_G']['which_model_G'] == 'uni_lora_resnet_multimag':
        return UNI_lora_resnet_MultiMag(opt['network_G']['out_nc'])
    if opt['network_G']['which_model_G'] == 'uni_lora_multimag_attn':
        return UNI_lora_cls_MultiMag_Attn(opt['network_G']['out_nc'])
    
    elif opt['network_G']['which_model_G'] == 'vit_baseline':
        return ViT_baseline(opt['network_G']['out_nc'])
    elif opt['network_G']['which_model_G'] == 'resnet_baseline':
        return ResNet_baseline(opt['network_G']['out_nc'])
    elif opt['network_G']['which_model_G'] == 'mobilenetv2_baseline':
        return MobileNetV2_baseline(opt['network_G']['out_nc'])
    elif opt['network_G']['which_model_G'] == 'convnextv2_baseline':
        return ConvNeXtV2_baseline(opt['network_G']['out_nc'])
    elif opt['network_G']['which_model_G'] == 'swinv2_baseline':
        return ConvNeXtV2_baseline(opt['network_G']['out_nc'])
    
    elif opt['network_G']['which_model_G'] == 'unet_baseline':
        return UNet(opt['network_G']['out_nc'])
    elif opt['network_G']['which_model_G'] == 'segformer_baseline':
        return Segformer(opt['network_G']['out_nc'])
    
    elif opt['network_G']['which_model_G'] == 'transformer_reorder':
        return TransformerReorder()
    elif opt['network_G']['which_model_G'] == 'stacked_model':
        return StackedModel(opt['network_G']['out_nc'])
    elif opt['network_G']['which_model_G'] == 'stacked_bbdm':
        return StackedDiffusionModel(opt)
    elif opt['network_G']['which_model_G'] == 'discrete_stacked_bbdm':
        return DiscreteStackedDiffusionModel(opt)
    else:
        raise NotImplementedError('Model [{:s}] is not recognized.'.format(opt['network_G']['which_model_G']))