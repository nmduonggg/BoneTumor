from __future__ import absolute_import
# from .mobilenetv2 import mobilenetv2
from .GigaPath import GigaPath
from .UNI import UNI
from .unet import UNet
from .UNI_lora import UNI_lora
from .UNI_lora_cls import UNI_lora_cls

def create_model(opt):
    if opt['network_G']['which_model_G'] == 'gigapath':
        return GigaPath(opt['network_G']['out_nc'])
    elif opt['network_G']['which_model_G'] == 'uni':
        return UNI(opt['network_G']['out_nc'])
    elif opt['network_G']['which_model_G'] == 'unet':
        return UNet(opt['network_G']['out_nc'])
    elif opt['network_G']['which_model_G'] == 'uni_lora':
        return UNI_lora(opt['network_G']['out_nc'])
    elif opt['network_G']['which_model_G'] == 'uni_lora_cls':
        return UNI_lora_cls(opt['network_G']['out_nc'])
    else:
        raise NotImplementedError('Model [{:s}] is not recognized.'.format(opt['network_G']['which_model_G']))