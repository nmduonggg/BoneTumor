from __future__ import absolute_import
# from .mobilenetv2 import mobilenetv2
from .UNI_lora_cls import UNI_lora_cls
from .Transformer_reorder import TransformerReorder

def create_model(opt):
    if opt['network_G']['which_model_G'] == 'uni_lora_cls':
        return UNI_lora_cls(opt['network_G']['out_nc'])
    elif opt['network_G']['which_model_G'] == 'transformer_reorder':
        return TransformerReorder(opt['network_G']['out_nc'])
    else:
        raise NotImplementedError('Model [{:s}] is not recognized.'.format(opt['network_G']['which_model_G']))