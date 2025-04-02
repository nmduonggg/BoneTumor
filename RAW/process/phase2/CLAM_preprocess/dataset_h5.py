import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
import threading
import queue

from PIL import Image
import h5py
import argparse
from tqdm import tqdm
import json
import openslide

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi_path,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = openslide.open_slide(wsi_path)
		self.roi_transforms = img_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		if self.roi_transforms: img = self.roi_transforms(img)

		return {'img': img, 'coord': coord}

class WSI_Processor():
    def __init__(self, wsi_dataset, patch_dir, 
                 result_dir, global_id, infor_dict = None):
        super().__init__()
        self.wsi_dataset = wsi_dataset
        self.patch_dir = patch_dir
        self.result_dir = result_dir
        self.infor_dict = infor_dict
        
        self.img_folder = os.path.join(self.result_dir,self.infor_dict['case'], "patches")
        os.makedirs(self.img_folder, exist_ok=True)
        
        self.global_id = global_id
        self.metadata_queue = queue.Queue()
        
    def save_img(self, img, path):
        img.save(path)
        return
    
    def process(self):
        metadatas = []
        
        for id, data in tqdm(enumerate(self.wsi_dataset), total=len(self.wsi_dataset)):
            img = data['img']
            coord = data['coord']
            
            img_path = os.path.join(self.img_folder, f"incase_{self.global_id}.jpg")
            
            img.save(img_path)
            metadata = {
                'coord': coord,
                'inslide_id': id,
                'incase_id': self.global_id
            }
            metadata.update(self.infor_dict)
            metadatas.append(metadata)
            
            self.global_id += 1
            
        return metadatas

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--DATA_DIR", type=str, help="data directory")
    args.add_argument("--PATCH_DIR", type=str, help="patches dir, to get coords")
    args.add_argument("--RESULT_DIR", type=str, help="dir where to store results (cropped images and metadata)")
     
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
    coord_dir = args.PATCH_DIR
    result_dir = args.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)
    
    slides = [slide for slide in os.listdir(data_dir) if (".mrxs" in slide) and (os.path.isfile(os.path.join(data_dir, slide)))]
    
    global_id = 0
    metadatas = []
    for slide in slides:
        print("===", slide, "===")
        wsi_path = os.path.join(data_dir, slide)
        coord_fn = slide.replace(".mrxs", ".h5")
        coord_path = os.path.join(coord_dir, coord_fn)
        if not os.path.isfile(coord_path): continue
        
        infor_dict = {
            "case": os.path.basename(data_dir),
            "slide_name": slide
        }
        wsi_bag = Whole_Slide_Bag_FP(coord_path, wsi_path) 
        wsi_processor = WSI_Processor(wsi_bag, coord_dir, result_dir, global_id, infor_dict)
        
        # process wsi bag and get final global id
        slide_metadatas = wsi_processor.process()
        metadatas += slide_metadatas
        global_id = wsi_processor.global_id
        
        
    # saving
    json_path = os.path.join(result_dir, infor_dict['case'], 'metadata.json')
    data2json(metadatas, json_path)
    
    return

if __name__ == '__main__':
    main_1_case()
        
        