import os
import numpy as np
import pandas as pd
from icecream import ic

from torch.utils.data import Dataset
from torchvision import transforms
import threading
import queue

from PIL import Image
import h5py
import argparse
from tqdm import tqdm
import json
from openslide import open_slide

# Image.MAX_IMAGE_PIXELS = 933120000
Image.MAX_IMAGE_PIXELS = None

def gt_name_from_slide_name(slide_name, gt_dir):
    assert (".mrxs" in slide_name), "Invalid slide name"
    gt_name1 = slide_name.replace(".mrxs", "-x8-labels")
    gt_name2 = slide_name.replace(".mrxs", "-labels")
    
    for ext in [".png", ".jpg"]:
        if os.path.isfile(os.path.join( gt_dir, gt_name1 + ext )):
            scale = 3
            return gt_name1 + ext, scale
        elif os.path.isfile(os.path.join( gt_dir, gt_name2 + ext )):
            scale = 2
            return gt_name2 + ext, scale
    
    assert (0)

class GT_Bag_FP(Dataset):
    def __init__(self, metadata, gt_dir, slide_wsi, gt, scale,
        patch_size=1024, img_transforms=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            img_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.gt_dir = gt_dir
        
        # self.gt = self.gt.resize([int(s / 2**self.scale) for s in self.gt.size], resample=Image.NEAREST)
        
        self.roi_transforms = img_transforms
        self.metadata = metadata
        self.scale = scale
        
        wsi = slide_wsi
        self.offset_dict = {
            'offset_x': int(wsi.properties['openslide.bounds-x']),
            'offset_y': int(wsi.properties['openslide.bounds-y']),
            'bounds_width': int(wsi.properties['openslide.bounds-width']),
            'bounds_height': int(wsi.properties['openslide.bounds-height'])
        }
        self.wsi_dim = wsi.dimensions
        
        # get scale and name
        self.gt = gt
        
        # scale has been initialized
        self.patch_size = patch_size
        self.offset_dict['ori_patch_size'] = self.patch_size
        self.offset_dict['scale'] = self.scale
        
        self.patch_size = np.round(self.patch_size / 2**self.scale)
            
    def __len__(self):
        return 1
            
    def __getitem__(self, idx):
        coord = np.array(self.metadata['coord'], dtype=int)
        
        lu = coord - np.array([self.offset_dict['offset_x'], self.offset_dict['offset_y']])
        
        lu = np.round(lu / 2**self.scale)
        rb = lu + np.array([self.patch_size, self.patch_size]).astype(int)
        bbox = [lu[0], lu[1], rb[0], rb[1]]
        # img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        img = self.gt.crop(bbox).convert("RGB")
        
        slide_incase_id = self.metadata['incase_id']
        slide_inslide_id = self.metadata['inslide_id']

        return {
            'img': img, 'coord': coord, 'rs_coord': lu, 
            'incase_id': slide_incase_id, 'inslide_id': slide_inslide_id}

class GT_Processor():
    def __init__(self, wsi_dataset,  
                result_dir, global_id, infor_dict = None):
        super().__init__()
        self.wsi_dataset = wsi_dataset
        # self.patch_dir = patch_dir
        self.result_dir = result_dir
        self.case = wsi_dataset.metadata['case']
        self.scale = wsi_dataset.scale
        
        self.img_folder = os.path.join(self.result_dir, infor_dict['case'], "patches")
        os.makedirs(self.img_folder, exist_ok=True)
        
        # self.global_id = global_id
        self.metadata_queue = queue.Queue()
        
    def save_img(self, img, path):
        img.save(path)
        return
    
    def process(self):
        metadatas = []
        
        data = self.wsi_dataset[0]
        img = data['img']
        coord = data['coord']
        rs_coord = data['rs_coord']
        incase_id = data['incase_id']
        
        img = img.resize([int(s * 2**self.scale) for s in img.size], resample=Image.NEAREST)
        
        img_path = os.path.join(self.img_folder, f"incase_{incase_id}.png")
        
        img.save(img_path)
        metadata = {
            'original_coord': coord,
            'resized_coord': rs_coord,
            'inslide_id': data['inslide_id'],
            'incase_id': incase_id,
        }
        metadata.update(self.wsi_dataset.offset_dict)
        metadatas.append(metadata)
        
        return metadatas
    
    # def process(self):
    #     metadatas = []
        
    #     for id, data in tqdm(enumerate(self.wsi_dataset), total=len(self.wsi_dataset)):
    #         img = data['img']
    #         coord = data['coord']
    #         rs_coord = data['rs_coord']
    #         incase_id = data['incase_id']
            
    #         img = img.resize([int(s * 2**self.scale) for s in img.size])
            
    #         img_path = os.path.join(self.img_folder, f"incase_{incase_id}.jpg")
            
    #         img.save(img_path)
    #         metadata = {
    #             'original_coord': coord,
    #             'resized_coord': rs_coord,
    #             'inslide_id': data['inslide_id'],
    #             'incase_id': incase_id,
    #             'patch_size': self.patch_size
    #         }
    #         metadata.update(self.infor_dict)
    #         metadatas.append(metadata)
            
    #         # self.global_id += 1
            
    #     return metadatas

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--DATA_DIR", type=str, help="data directory")
    args.add_argument("--METADATA_PATH", type=str, help="metadata path for reference")
    args.add_argument("--SLIDE_DIR", type=str, help="slide dir, to get original image")
    args.add_argument("--RESULT_DIR", type=str, help="dir where to store results (cropped images and metadata)")
    args.add_argument("--patch_size", type=int)
     
    return args.parse_args()

def data2json(data, json_path):
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)
    return

def main_1_case():
    args = parse_args()
    
    data_dir = args.DATA_DIR
    slide_dir = args.SLIDE_DIR
    metadata_path = args.METADATA_PATH
    result_dir = args.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)
    
    with open(metadata_path, 'r') as f:
        slide_metadatas = json.load(f)
    
    global_id = 0
    metadatas = []
    num_metadata = len(slide_metadatas)
    last_slide_fn, gt, slide_wsi = None, None, None
    for idx in tqdm(range(num_metadata), total=num_metadata):
        mtd = slide_metadatas[idx]
        slide_fn = mtd['slide_name']
        
        gt_name, scale = gt_name_from_slide_name(slide_fn, data_dir)
        gt_path = os.path.join(data_dir, gt_name)
        
        slide_path = os.path.join(slide_dir, slide_fn)
        if slide_fn != last_slide_fn or last_slide_fn is None:
            print("open new slide:", slide_fn)
            slide_wsi = open_slide(slide_path)
            last_slide_fn = slide_fn
            
            gt = Image.open(gt_path).convert("RGB")

            
        infor_dict = {
            "case": os.path.basename(data_dir),
            "slide_name": slide_fn
        }
        
        wsi_bag = GT_Bag_FP(mtd, data_dir, slide_wsi, gt, scale, patch_size=args.patch_size) 
        wsi_processor = GT_Processor(wsi_bag, result_dir, global_id, infor_dict)
        
        # process wsi bag and get final global id
        gt_metadatas = wsi_processor.process()
        metadatas += gt_metadatas
        # global_id = wsi_processor.global_id
        
    # saving
    json_path = os.path.join(result_dir, infor_dict['case'], 'metadata.json')
    data2json(metadatas, json_path)
    
    return

if __name__ == '__main__':
    main_1_case()
        
        