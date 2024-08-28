import os
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
import PIL

import options.options as option
from data import create_dataloader, create_dataset
from model import create_model
import utils.utils as utils
import data.utils as data_utils
import matplotlib.patches as mpatches

from huggingface_hub import login

# login("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")  # UNI

login("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")  # Prov-GigaPath

# Prov-Gigapath
# os.environ['HF_TOKEN'] = "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"

abspath = os.path.abspath(__file__)
PIL.Image.MAX_IMAGE_PIXELS = None

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
current_infer_img = os.path.basename(os.path.dirname(args.infer_dir))
working_dir = os.path.join('.', 'infer', opt['name'], current_infer_img)
os.makedirs(working_dir, exist_ok=True)
print("Working dir: ", working_dir)
    
model = create_model(opt)

# fix and load weight
# state_dict = torch.load(opt['path']['pretrain_model'], map_location='cpu')

if args.weight_path != '':
    state_dict = torch.load(args.weight_path, map_location='cpu')
    # current_dict = model.state_dict()
    # new_state_dict={k:v if v.size()==current_dict[k].size()  else  current_dict[k] for k,v in zip(current_dict.keys(), state_dict.values())}    # fix the size of checkpoint state dict
    _strict = False

    model.load_state_dict(state_dict, strict=_strict)  
    print("[INFO] Load weight from:", args.weight_path)
else:
    print("No pretrained weight found")
    
# Init
crop_sz = 512
step = 512
infer_size = 512
color_map = [
    (0, 0, 255),    # non-tumor - blue
    (0, 255, 0),    # viable - green
    (255, 0, 0),    # non-viable - red
    (255, 255, 255)
]

def prepare():
    global crop_sz, step
    # img = cv2.imread(args.infer_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Loading inference image...")
    with open(os.path.join(args.infer_dir, '_size.txt'), 'r') as f:
        rows, cols, h, w = [int(x) for x in f.read().split(' ')]
        
    patches_list = []
    for i in range(rows):
        for j in range(cols):
            full_fn = os.path.join(args.infer_dir, "%d_%d.npy" % (j, i))
            patches_list.append(full_fn)
        
    return (patches_list, rows, cols, h, w)
    
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
    
    idx += 1 # [0, 1, 2, 3]
    class_tensor = torch.ones_like(patch) * torch.tensor(idx).reshape(1, 1, 1, 1)
    class_np = class_tensor.squeeze(0).permute(1,2,0).numpy() 
    
    return color_np, class_np

def alpha_blending(im1, im2, alpha):
    out = cv2.addWeighted(im1, alpha , im2, 1-alpha, 0)
    return out

def infer():
    global color_map
    
    model.to(device)
    model.eval()
    
    preprocess_elems = prepare()
    patches_list, num_h, num_w, h, w = preprocess_elems
    kwargs = {
        'sr_list': patches_list,
        'num_h': num_h,
        'num_w': num_w,
        'h': h, 'w': w,
        'patch_size': crop_sz, 
        'step': step
    }
    img = (postprocess(**kwargs) * 255).astype(np.uint8)
        
    patches_dir = os.path.join(working_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    preds_list, class_list = [], []
    feature_list = None
    
    bs = 256
    num_batches = len(patches_list) // bs + 1
    
    all_imgs = []
    
    for batch_id in tqdm(range(num_batches), total=num_batches):
        current_patches = patches_list[batch_id*bs: min(batch_id*bs + bs, len(patches_list))]
        batched_patches = []
        for patch in current_patches:
    
    # for i, patch in tqdm(enumerate(patches_list), total=len(patches_list)):
        
            if 'txt' in patch: continue
            patch = np.load(patch, allow_pickle=True)
            bg = np.ones((crop_sz, crop_sz, 3), 'float32') * 255
            r, c, _ = patch.shape
            bg[:r, :c, :] = patch
            patch = bg.astype(np.uint8)
            # patch = cv2.resize(patch, (infer_size, infer_size))
            
            ori_patch = copy.deepcopy(patch)
            edge_score = laplacian_score(patch).mean()
            # patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            
            patch = data_utils.normalize_np(patch / 255.0).astype(np.float32)
            patch_tensor = torch.tensor(patch).permute(2,0,1)
            batched_patches.append(patch_tensor)
            # im = patch_tensor.unsqueeze(0).to(device)
        
        # if edge_score <= 1: # filter background
        #     pred, class_ = index2color(-1, im.cpu(), color_map)
        #     preds_list.append(pred)   # skip
        #     class_list.append(class_)
        #     continue
        
        # plt.imsave(os.path.join(patches_dir, f'./patch_{i}.png'), ori_patch)
        im = torch.stack(batched_patches).to(device)
        all_imgs.append(im.detach().cpu())
        
        with torch.no_grad():
            feature = model.encode(im)
        feature = feature.clone().detach()
        feature_list = feature if feature_list is None else torch.cat([feature_list, feature], dim=0)
    all_imgs = torch.cat(all_imgs, dim=0)
    centroids, labels = utils.clustering_pytorch(feature_list, 4)
    for i in range(len(patches_list)):
        pred = labels[i].long().item()
        tim = all_imgs[i:i+1, ...].cpu()
        # print(i, i+1, len(patches_list))
        # print(tim.shape)
        pred, class_ = index2color(pred, tim, color_map)
        class_list.append(class_)
        preds_list.append(pred)
        
    fx = 5e-2
    fy = 5e-2
    img = cv2.resize(img, None, fx=fx, fy=fy)
    plt.imsave(os.path.join(working_dir, 'original_img.png'), img)    
    
    kwargs['sr_list'] = preds_list
    # print(*preds_list)
    prediction = (postprocess(**kwargs) * 255).astype(np.uint8)
    prediction = cv2.resize(prediction, None, fx=fx, fy=fy)
    # cut to align img and prediction
    # img = img[:h, :w, :]
    blend_im = alpha_blending(
        img, prediction, 0.6)
    
    del prediction  # free mem
    
    # binary_im = laplacian_score(img)
    # binary_im = (binary_im > 0).astype(np.float32)
    # binary_im = np.expand_dims(binary_im, axis=-1)
    # # binary_im = cv2.resize(binary_im, None, fx=fx, fy=fy)
    # total_pixels = binary_im.sum()
    
    # blend_im = (blend_im * binary_im).astype(np.uint8)
    
    kwargs['sr_list'] = class_list
    class_prediction = (postprocess(**kwargs)).astype(np.uint8)
    class_prediction = class_prediction.astype(np.uint8)
    total_pixels = class_prediction.shape[0] * class_prediction.shape[1]
    class_im = [
        (class_prediction==i).astype(int).sum() / 3 for i in range(0, 4)
    ]
    class_percent = [cl / total_pixels for cl in class_im]
    
    # del binary_im   # free mem

    # prediction = cv2.resize(prediction, None, fx=fx, fy=fy)
    # blend_im = cv2.resize(blend_im, None, fx=fx, fy=fy)
    
    # plt.imsave(os.path.join(working_dir, 'out_giga.png'), prediction)
    # plt.imsave(os.path.join(working_dir, 'binary_img.png'), binary_im)
    
    
    # Final result
    plt.figure(figsize=(16, 10))
    plt.imshow(blend_im)
    # create a patch (proxy artist) for every color 
    labels = ['non-tumor', 'viable', 'non-viable', 'background']
    patches = [ mpatches.Patch(color=np.array(color_map[i]) / 255.,
                               label=f"{labels[i]} - ({round(class_percent[i], 2)})") for i in range(len(labels)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0. )
    plt.axis("off")
    plt.savefig(os.path.join(working_dir, 'blend_giga.png'))
    plt.cla()
        
    print("[RESULT] Each class non-tumor, viable, non-viable:", class_percent)
    
    return

infer()
    
    
        
        
        
        
        