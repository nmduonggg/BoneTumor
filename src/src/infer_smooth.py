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

from torchvision import transforms
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
working_dir = os.path.join('.', 'infer', opt['name'], 'smooth', current_infer_img)
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
step = 240
infer_size = 256
small_h = small_w = 16
small_step = step // (crop_sz / small_h)
color_map = [
    [255, 255, 255],    # background
    [0, 128, 0],    # Viable tumor
    [255, 143, 204],    # Necrosis
    [255, 0, 0],    # Fibrosis/Hyalination
    [0, 0, 0],  # Hemorrhage/ Cystic change
    [165, 42, 42],  # Inflammatory
    [0, 0, 255]]    # Non-tumor tissue

def apply_threshold_mapping(image):
    # Create masks for pixels that are closer to green or pink
    # Initialize the output image with the original image
    tolerance = 50
    
    output = np.ones_like(image)*255 # 2D only
    masks = []
    for idx, color in enumerate(color_map):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        output[mask] = color
        # output[mask] = idx

    return output

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
    
    # patch = F.interpolate(patch, (crop_sz, crop_sz))
    
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
    
    with open(os.path.join(working_dir, "result.txt"), "a") as f:
        f.write(f"{infer_name}\n")
    
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
        
        # ori_patch = copy.deepcopy(patch)
        edge_score = laplacian_score(patch).mean()
        # patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        
        # normalize
        # patch = data_utils.normalize_np(patch / 255.0).astype(np.float32)
        # patch_tensor = torch.tensor(patch).permute(2,0,1)
        # im = patch_tensor.unsqueeze(0).to(device)
        
        transform = transforms.Compose(
                [
                    # transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        im = transform(patch).float().unsqueeze(0)
        
        if edge_score <= 5: # filter background
            # pred, class_ = index2color(0, im.cpu(), color_map)
            pred_im = np.zeros((small_h, small_w, 7))
            pred_im[:, :, 0] = 0.5
            preds_list.append(pred_im)   # skip
            # class_list.append(class_)
            continue
        
        im = im.to(device)
        with torch.no_grad():
            pred = model(im)
        # pred, class_ = index2color(torch.argmax(pred.cpu().detach(), dim=-1), im.cpu(), color_map)
        pred = pred.cpu().squeeze(0).numpy()
        pred = np.ones((small_h, small_w, 7)) * pred
        
        preds_list.append(pred)
        # class_list.append(class_)
        # try:
        #     class_counts[class_] += 1
        # except:
        #     print(class_)
    
    kwargs['sr_list'] = preds_list
    kwargs['channel'] = 7
    kwargs['step'] = int(small_step)
    kwargs['patch_size'] = small_h
    prediction = postprocess(**kwargs)  # hxwx7
    # kwargs['sr_list'] = class_list
    # class_prediction = (postprocess(**kwargs)).astype(np.uint8)
    prediction = np.argmax(prediction, axis=-1)
    output = np.zeros((prediction.shape[0], prediction.shape[1], 3))
    
    for i in range(len(color_map)):
        color = np.array(color_map[i])
        mask = np.all(prediction==i, axis=-1)
        output[mask] = color
        
        class_counts[i] += mask.sum() 
    
    prediction = cv2.resize(output, (img.shape[1], img.shape[0]))
    
    
    # cut to align img and prediction
    # img = img[:h, :w, :]
    img_mask = np.expand_dims((laplacian_score(img) > 5), axis=2)
    prediction = prediction * img_mask + np.ones_like(prediction)*255*(1-img_mask)
    
    # blend_im = alpha_blending(
    #     img, prediction, 0.6)
    
    
    plt.imsave(os.path.join(working_dir, f"{infer_name.split('.')[0]}_pred.png"), prediction) 
    
    
    
    del prediction  # free mem
    
    # class_counts = np.array(class_counts)
    class_percent = np.array(class_counts) / np.sum(np.array(class_counts))
    print(class_percent)
    
    labels = ['background', 'viable', 'necrosis', 'fibrosis/hyalination', 'hemorrhage/cystic-change', 'inflammatory', 'non-tumor']

    # plt.imsave(os.path.join(working_dir, f"{infer_name.split('.')[0]}_blend.png"),
    #            blend_im.astype(np.uint8))
    print("Save prediction done")
        
    print("[RESULT]")
    for i in range(7):
        print(f"{labels[i]} - {round(class_percent[i], 4)}")
        with open(os.path.join(working_dir, "result.txt"), "a") as f:
            f.write(f"{labels[i]} - {round(class_percent[i], 4)}\n")
    with open(os.path.join(working_dir, "result.txt"), "a") as f:
        f.write(f"-------------\n")
        
    print("-"*20)
    
    return class_counts


if __name__=='__main__':
    
    case_patch_counts = [0 for _ in range(len(color_map))]
    with open(os.path.join(working_dir, "result.txt"), "w") as f:
        f.write(f"======={args.infer_dir}=======\n")

    for infer_path in os.listdir(args.infer_dir):
        
        # start
        infer_path = os.path.join(args.infer_dir, infer_path)
            
        print("Process: ", infer_path)
        slide_patch_counts = infer(infer_path)
        print(slide_patch_counts)
        
        for i, class_count in enumerate(slide_patch_counts):
            case_patch_counts[i] += slide_patch_counts[i]
            
    case_patch_percents = np.array(case_patch_counts) / np.sum(np.array(case_patch_counts))
    with open(os.path.join(working_dir, "result.txt"), "a") as f:
        f.write(f"{case_patch_percents}")
        
        
            
            
            
            
            