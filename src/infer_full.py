import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from torchvision import transforms
import options.options as option
from model import create_model
import utils.utils as utils

from huggingface_hub import login

abspath = os.path.abspath(__file__)


parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YAML file.')
parser.add_argument('-root', type=str, default=None, choices=['.'])
parser.add_argument('--labels_dir', type=str, required=True)
parser.add_argument('--images_dir', type=str, required=True)
parser.add_argument('--phase1_path', type=str, required=True)
parser.add_argument('--phase2_path', type=str, required=True)

args = parser.parse_args()
opt = option.parse(args.opt, root=args.root)

opt = option.dict_to_nonedict(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % opt['gpu_ids'][0]
device = torch.device('cuda:0' if opt['gpu_ids'] is not None else 'cpu')

# HF Login to get pretrained weight
login(opt['token'])
    
model = create_model(opt)

# fix and load weight
# state_dict = torch.load(opt['path']['pretrain_model'], map_location='cpu')

if args.phase1_path != '' and args.phase2_path != '':
    phase1_dict = torch.load(args.phase1_path, map_location='cpu')
    phase2_dict = torch.load(args.phase2_path, map_location='cpu')
    _strict = True

    model.load_state_dict(phase1_dict, phase2_dict, strict=_strict)  
    print(f"[INFO] Load phase 1 weight from: {args.phase1_path}")
    print(f"[INFO] Load phase 2 weight from: {args.phase2_path}")
else:
    print("No pretrained weight found")
    
# Init
crop_sz = 256*8
step = 256*8
small_h = small_w = 256
ratio = int(crop_sz / small_h)
small_step = step // ratio
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

def read_percent_from_color(image):
    tolerance = 50
    masks = []
    for idx, color in enumerate(color_map):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        masks.append(mask.sum())

    return masks

def open_img(image_path):
    img = cv2.imread(image_path)[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def prepare(infer_path):
    global crop_sz, step
    
    img = open_img(infer_path)
    
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

def infer(infer_path, label_path):
    global color_map
    infer_name = os.path.basename(infer_path)
    
    with open(os.path.join(working_dir, "result.txt"), "a") as f:
        f.write(f"{infer_name}\n")
    
    fx = 5e-2
    fy = 5e-2
    
    # label = open_img(label_path)
    # label_mask = np.any(np.abs(label - np.array([255, 255, 255])) > 5, axis=-1).astype(int)
    # label_mask = cv2.resize(label_mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    
    # del label
    
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
    # for k, v in kwargs.items():
    #     if k != 'sr_list': print(k, v)
    # img = postprocess(**kwargs)
    img = img[:h,:w, :]
    img = cv2.resize(img, None, fx=fx, fy=fy)
    print(img.max())
    plt.imsave(os.path.join(working_dir, infer_name), img) 
    print("Save original image done") 
    
    preds_list, class_list = [], []
    class_counts = [0 for _ in range(7)]
    
    for i, patch in tqdm(enumerate(patches_list), total=len(patches_list)):
        
        bg = np.ones((crop_sz, crop_sz, 3), 'float32') * 255
        r, c, _ = patch.shape
        bg[:r, :c, :] = patch
        patch = bg.astype(np.uint8)
        
        edge_score = laplacian_score(patch).mean()
        
        # normalize
        transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        im = transform(patch).float().unsqueeze(0)
        
        if edge_score <= 5: # filter background
            pred_im = np.zeros((crop_sz, crop_sz, 7))
            pred_im[:, :, 0] = 1e-9
            preds_list.append(pred_im)   # skip
            continue
        
        im = im.to(device)
        with torch.no_grad():
            pred = model(im)[..., :-1]
        # print(pred.shape)
        # print(pred.shape)
        pred = torch.softmax(pred, dim=-1).cpu().squeeze(0).numpy()
        # pred = np.ones((small_h, small_w, 7)) * pred
        
        preds_list.append(pred)
    
    kwargs['sr_list'] = preds_list
    kwargs['channel'] = 7
    
    prediction = postprocess(**kwargs)  # hxwx7
    prediction = np.expand_dims(np.argmax(prediction, axis=-1), axis=2)
    output = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype='uint8')
    
    for i in range(len(color_map)):
        color = np.array(color_map[i]).astype(np.uint8)
        mask = np.all(np.abs(prediction-i) < 1e-9, axis=-1)
        print(mask.sum())
        output[mask] = color
        
    prediction = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # label_mask = cv2.resize(label_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
    
    
    # cut to align img and prediction
    img_mask = np.expand_dims((np.abs(img[:, :, -1] - 255)) > 0, axis=-1)
    # label_mask = np.expand_dims(label_mask, axis=-1)
    
    prediction = prediction * img_mask + np.ones_like(prediction)*255*(1-img_mask)
    # prediction = prediction * label_mask + np.ones_like(prediction)*255*(1-label_mask)
    prediction = prediction.astype(np.uint8)
    
    class_counts = read_percent_from_color(prediction)
    
    blend_im = alpha_blending(
        img, prediction, 0.6)
    
    plt.imsave(os.path.join(working_dir, f"{infer_name.split('.')[0]}_pred.png"), prediction) 
    
    del prediction  # free mem
    
    class_counts = class_counts[1:] # skip_background
    class_percent = np.array(class_counts) / (np.sum(np.array(class_counts)) + 1e-8)
    print(class_percent)
    
    labels = ['background', 'viable', 'necrosis', 'fibrosis/hyalination', 'hemorrhage/cystic-change', 'inflammatory', 'non-tumor']

    plt.imsave(os.path.join(working_dir, f"{infer_name.split('.')[0]}_blend.png"),
               blend_im.astype(np.uint8))
    print("Save prediction done")
        
    print("[RESULT]")
    for i in range(6):
        # if i==0: continue
        # i += 1
        print(f"{labels[i+1]} - {round(class_percent[i], 4)}")
        with open(os.path.join(working_dir, "result.txt"), "a") as f:
            f.write(f"{labels[i+1]} - {round(class_percent[i], 3)*100}%\n")
            
    huvos_ratio = 1 - class_counts[0] / np.sum(class_counts[:5]) 
    
    if class_percent[-1] >= 0.99:
        huvos_ratio = None
    
    with open(os.path.join(working_dir, "result.txt"), "a") as f:
        if huvos_ratio is not None:
            huvos_ratio = round(huvos_ratio*100, 1)
            f.write(f'total_necrosis: {huvos_ratio}% \n')
        else:
            f.write('total_necrosis: N/A \n')
        f.write(f"-------------\n")
        
    print("-"*20)
    
    return class_counts, huvos_ratio

def huvos_classify(huvos_ratio):
    labels = ["I", "II", "III", "IV"]
    index = 0
    if huvos_ratio < 50:
        index = 0
    elif 50 <= huvos_ratio < 90:
        index = 1
    elif 90 <= huvos_ratio < 100:
        index = 2
    else:
        index = 3
    return labels[int(index)]


if __name__=='__main__':
    
    
    done_cases = [f"Case_{n}" for n in [2]]
    
    for case in os.listdir(args.labels_dir):
        if case in done_cases: continue
        # if case != 'Case_1': continue
        
        label_dir = os.path.join(args.labels_dir, case)
        image_dir = os.path.join(args.images_dir, case)
        
        # initialize working dir for each case
        working_dir = os.path.join('.', 'infer', 'smooth_stacked', opt['name'], case)
        os.makedirs(working_dir, exist_ok=True)
        print("Working dir: ", working_dir)
    
        case_patch_counts = [0 for _ in range(len(color_map)-1)]
        with open(os.path.join(working_dir, "result.txt"), "w") as f:
            f.write(f"======={case}=======\n")
            
        huvos_case = []
        
        label_names = [n for n in os.listdir(label_dir) if ('.jpg' in n or '.png' in n)]

        for label_name in label_names:
            
            # start
            
            if "x8" in label_name:
                image_name = label_name.split("-x8")[0] + '.png'
                upsample = True
            else:
                image_name = label_name.split("-labels")[0] + '.png'
            
            infer_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, label_name)
                
            print("Process: ", infer_path)
            slide_patch_counts, huvos_ratio = infer(infer_path, label_path)
            if huvos_ratio is not None:
                huvos_case.append(huvos_ratio)
            
            for i, class_count in enumerate(slide_patch_counts):
                case_patch_counts[i] += slide_patch_counts[i]
                
        case_patch_percents = np.array(case_patch_counts) / np.sum(np.array(case_patch_counts))
        
        huvos_case = np.mean(huvos_case)
        huvos_case = round(huvos_case*100, 1)
        with open(os.path.join(working_dir, "result.txt"), "a") as f:
            f.write(f"Total Necrosis on case: {huvos_case}%\n")
            f.write(f"Huvos: {huvos_case}")
        
            
            
            
            
            