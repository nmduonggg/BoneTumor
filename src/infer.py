import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

import cv2
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import argparse
from tqdm import tqdm
from PIL import Image

import options.options as option
from data import create_dataloader, create_dataset
from model import create_model
import utils.utils as utils
import data.utils as data_utils
import matplotlib.patches as mpatches

from huggingface_hub import login

abspath = os.path.abspath(__file__)
# Image.MAX_IMAGE_PIXELS = (2e40).__str__()


parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YAML file.')
parser.add_argument('-root', type=str, default=None, choices=['.'])
parser.add_argument('--infer_dir', type=str, required=True)
parser.add_argument('--weight_path', type=str, required=True)
args = parser.parse_args()
opt = option.parse(args.opt, root=args.root)

opt = option.dict_to_nonedict(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % opt['gpu_ids'][0]
device = torch.device('cuda:0' if opt['gpu_ids'] is not None else 'cpu')

# current_infer_img = os.path.parent(args.infer_dir)[:-4]
current_infer_img = os.path.basename(args.infer_dir)
working_dir = os.path.join('.', 'infer', opt['name'], current_infer_img)
os.makedirs(working_dir, exist_ok=True)
print("Working dir: ", working_dir)

# HF Login to get pretrained weight
login(opt['token'])
    
model = create_model(opt)

# fix and load weight
# state_dict = torch.load(opt['path']['pretrain_model'], map_location='cpu')

if args.weight_path != '':
    state_dict = torch.load(args.weight_path, map_location='cpu')
    # current_dict = model.state_dict()
    # new_state_dict={k:v if v.size()==current_dict[k].size()  else  current_dict[k] for k,v in zip(current_dict.keys(), state_dict.values())}    # fix the size of checkpoint state dict
    _strict = True

    model.load_state_dict(state_dict, strict=_strict)  
    print("[INFO] Load weight from:", args.weight_path)
else:
    print("No pretrained weight found")
    
# Init
crop_sz = 256
step = 256
infer_size = 256
color_map = [
    [255, 255, 255],    # background
    [0, 128, 0],    # Viable tumor
    [255, 143, 204],    # Necrosis
    [255, 0, 0],    # Fibrosis/Hyalination
    [0, 0, 0],  # Hemorrhage/ Cystic change
    [165, 42, 42],  # Inflammatory
    [0, 0, 255]]    # Non-tumor tissue

def prepare(infer_path):
    global crop_sz, step
    
    # img = Image.open(infer_path).convert("RGB")
    img = cv2.imread(infer_path)[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.array(img)
    
    return [img, utils.crop(img, crop_sz, step)]
    
def postprocess(**kwargs):
    return utils.combine(**kwargs)

def laplacian_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    mask_img = cv2.convertScaleAbs(laplac)
    return mask_img

def index2color(idx, patch, color_map):
    
    patch = F.interpolate(patch, (crop_sz, crop_sz))
    
    color = color_map[idx]
    color_tensor = torch.ones_like(patch) * torch.tensor(color).reshape(1, -1, 1, 1)
    color_np = color_tensor.squeeze(0).permute(1,2,0).numpy() / 255.0
    
    # idx += 1 # [0, 1, 2, 3]
    class_tensor = torch.ones_like(patch) * torch.tensor(idx).reshape(1, 1, 1, 1)
    class_np = class_tensor.squeeze(0).permute(1,2,0).numpy() 
    
    return color_np, idx

def alpha_blending(im1, im2, alpha):
    out = cv2.addWeighted(im1, alpha , im2, 1-alpha, 0)
    return out

def infer(infer_path):
    global color_map
    infer_name = os.path.basename(infer_path)
    
    fx = 5e-2
    fy = 5e-2
    
    model.to(device)
    model.eval()
    
    img, preprocess_elems = prepare(infer_path)
    patches_list, num_h, num_w, h, w = preprocess_elems
    kwargs = {
        'sr_list': patches_list,
        'num_h': num_h,
        'num_w': num_w,
        'h': h, 'w': w,
        'patch_size': crop_sz, 
        'step': step
    }
    # img = postprocess(**kwargs)
    img = cv2.resize(img, None, fx=fx, fy=fy)
    print(img.max())
    plt.imsave(os.path.join(working_dir, infer_name), img) 
    print("Save original image done") 
        
    # patches_dir = os.path.join(working_dir, "patches")
    # os.makedirs(patches_dir, exist_ok=True)
    preds_list, class_list = [], []
    class_counts = [0 for _ in range(7)]
    for i, patch in tqdm(enumerate(patches_list), total=len(patches_list)):
        
        bg = np.ones((crop_sz, crop_sz, 3), 'float32') * 255
        r, c, _ = patch.shape
        bg[:r, :c, :] = patch
        patch = bg.astype(np.uint8)
        # patch = cv2.resize(patch, (infer_size, infer_size))
        
        ori_patch = copy.deepcopy(patch)
        edge_score = laplacian_score(patch).mean()
        # patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        
        # normalize
        patch = data_utils.normalize_np(patch / 255.0).astype(np.float32)
        patch_tensor = torch.tensor(patch).permute(2,0,1)
        im = patch_tensor.unsqueeze(0).to(device)
        
        if edge_score <= 5: # filter background
            pred, class_ = index2color(0, im.cpu(), color_map)
            preds_list.append(pred)   # skip
            class_list.append(class_)
            continue
        
        with torch.no_grad():
            pred = model(im)
        pred, class_ = index2color(torch.argmax(pred.cpu().detach(), dim=-1), im.cpu(), color_map)
        preds_list.append(pred)
        class_list.append(class_)
        try:
            class_counts[class_] += 1
        except:
            print(class_)
    
    kwargs['sr_list'] = preds_list
    prediction = (postprocess(**kwargs) * 255).astype(np.uint8)
    kwargs['sr_list'] = class_list
    # class_prediction = (postprocess(**kwargs)).astype(np.uint8)
    
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), fx=fx, fy=fy)
    
    
    # cut to align img and prediction
    # img = img[:h, :w, :]
    blend_im = alpha_blending(
        img, prediction, 0.6)
    
    plt.imsave(os.path.join(working_dir, f"{infer_name.split('.')[0]}_pred.png"), prediction) 
    
    del prediction  # free mem
    
    # binary_im = laplacian_score(img)
    # binary_im = (binary_im > 0).astype(np.float32)
    # binary_im = np.expand_dims(binary_im, axis=-1)
    # total_pixels = binary_im.sum()
    
    # blend_im = (blend_im * binary_im).astype(np.uint8)
    # class_prediction = (class_prediction * binary_im)
    
    # class_prediction = class_prediction.astype(np.uint8) * binary_im
    # class_list = np.array(class_list)
    # class_percent = [
    #     (class_list==i).astype(int).mean()  for i in range(7)
    # ]
    # class_percent = [cl / total_pixels for cl in class_im]
    class_counts = np.array(class_counts)
    class_percent = class_counts / np.sum(class_counts)
    print(class_percent)
    
    # del binary_im   # free mem
    # del class_prediction
    
    # binary_im = cv2.resize(binary_im, None, fx=fx, fy=fy)
    
    # prediction = cv2.resize(prediction, None, fx=fx, fy=fy)
  
    
    # blend_im = cv2.resize(blend_im, None, fx=fx, fy=fy)
    
    # plt.imsave(os.path.join(working_dir, 'out_giga.png'), prediction)
    # plt.imsave(os.path.join(working_dir, 'binary_img.png'), binary_im)
    
    # Final result
    # plt.figure(figsize=(16, 10))
    # plt.imshow(blend_im)
    # # create a patch (proxy artist) for every color 
    labels = ['background', 'viable', 'necrosis', 'fibrosis/hyalination', 'hemorrhage/cystic-change', 'inflammatory', 'non-tumor']
    # patches = [ mpatches.Patch(color=np.array(color_map[i]) / 255.,
    #                            label=f"{labels[i]} - ({round(class_percent[i], 2)})") for i in range(len(labels)) ]
    # # put those patched as legend-handles into the legend
    # plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0. )
    # plt.axis("off")
    # plt.savefig(os.path.join(working_dir, f'{infer_name.split(".")[0]}_blend.png'))
    # plt.cla()
    
    plt.imsave(os.path.join(working_dir, f"{infer_name.split('.')[0]}_blend.png"),
               blend_im.astype(np.uint8))
    print("Save prediction done")
        
    print("[RESULT]")
    for i in range(7):
        print(f"{labels[i]} - {round(class_percent[i], 4)}")
        
    print("-"*20)
    
    return


for infer_path in os.listdir(args.infer_dir):
    infer_path = os.path.join(args.infer_dir, infer_path)
    print("Process: ", infer_path)
    infer(infer_path)
    
    
        
        
        
        
        