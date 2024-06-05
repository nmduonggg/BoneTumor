import os
import numpy as np
import cv2


def main():
    train_folder = ['./Training-Set-1', './Training-Set-2']
    dst_folder = './Images'
    os.makedirs(dst_folder, exist_ok=True)

    cnt = 0
    for folder in train_folder:
        for set_folder in os.listdir(folder):
            print(f"Process {folder}/{set_folder}...")
            
            set_folder_path = os.path.join(folder, set_folder)
            for fn in os.listdir(set_folder_path):
                src_path = os.path.join(set_folder_path, fn)
                
                dst_path = os.path.join(dst_folder, fn)
                if 'csv' in src_path: continue
                img = cv2.imread(src_path)
                try:
                    writing = cv2.imwrite(dst_path, img)
                except:
                    print(src_path)
                assert(writing), f"Write image {src_path} fail ! {cnt} images written"
                
                cnt += 1

if __name__=='__main__':
    main()
