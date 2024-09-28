import os
import cv2

source_folder = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/process/RAW/REAL_WSIs/training_data_256/labels'
target_folder = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/training_data_256/labels'

cnt = 0
for img_name in os.listdir(source_folder):
    if cnt % 1000 == 0: print("Move ", cnt)
    img_path = os.path.join(source_folder, img_name)
    save_path = os.path.join(target_folder, img_name)
    img = cv2.imread(img_path)
    cv2.imwrite(save_path, img)
    
    cnt += 1