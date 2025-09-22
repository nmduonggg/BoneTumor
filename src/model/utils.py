from inspect import isfunction
import numpy as np
import cv2
import torch

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d



#### Cut and Combine ####
def crop_tensor(img, crop_sz, step):
    """
    im: [BxCxHxW]
    """
    n_channels = len(img.shape)
    if n_channels > 4:
        img = img.squeeze(0)
    h, w = img.shape[-2:]
    
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=[]
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            crop_img = img[:, :, x:x + crop_sz, y:y + crop_sz]
            
            lr_list.append(crop_img)
    h=x + crop_sz
    w=y + crop_sz
    return lr_list, num_h, num_w, h, w

def combine_output(sr_list, num_h, num_w, h, w, patch_size, step, channel=3):
    index=0
    sr_img = torch.zeros((channel, h, w)).to(sr_list[0].device)
    # print(h, w, num_h, num_w, channel)
    for i in range(num_h):
        for j in range(num_w):
            sr_subim = sr_list[index]
            
            sr_img[:, i*step: i*step+patch_size, j*step: j*step+patch_size]+=sr_subim
            index+=1
            
    # sr_img=sr_img.astype('float32')

    for j in range(1,num_w):
        sr_img[:, :, j*step:j*step+(patch_size-step)]/=2

    for i in range(1,num_h):
        sr_img[:, i*step:i*step+(patch_size-step),:]/=2
    return sr_img