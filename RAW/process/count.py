import os
import json

if __name__=='__main__':
    dts_folder = '/home/user01/aiotlab/nmduong/BoneTumor/RAW/REAL_WSIs/class_dict_256_retrain_for_phase2_with9.json'
    # image_folder = os.path.join(dts_folder, 'images')
    # cnt = 0
    with open(dts_folder, 'r') as f:
        data = json.load(f)
    for k in data.keys():
        print(k, len(data[k]))
        
    # print('Number of images:', cnt)