'''
Add new images of case/class_dict into current dataset_dict without overlapping
'''

import os
import json
import random

def random_split(indices, ratio1, ratio2):
    assert (ratio1 + ratio2 <= 1), "sum of ratios must <= 1"
    
    random.shuffle(indices)
    split_point1 = int(len(indices) * ratio1)
    split_point2 = int(len(indices) * (ratio1 + ratio2))
    set1 = indices[:split_point1]
    set2 = indices[split_point1: split_point2]
    set3 = indices[split_point2:]
    
    return set1, set2, set3

def write_to_json(train_indices, valid_indices, test_indices,
                  data = None,
                  path='./dataset_split.json'):
    if data is None:
        data = {
            'train': train_indices,
            'valid': valid_indices,
            'test': test_indices
        }
    with open(path, 'w') as f:
        json.dump(data, f)
    return

if __name__=='__main__':
    
    class_dict_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/class_dict_256.json'   # update continuously
    # case_dict_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/case_dict_256.json'
    old_class_dict_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/class_dict_256_bkup.json'
    dataset_dict_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/dataset_split_256_by_class_bkup.json'
    
    with open(class_dict_path, 'r') as f:
        case_dict = json.load(f)
    with open(old_class_dict_path, 'r') as f:
        old_case_dict = json.load(f)
    with open(dataset_dict_path, 'r') as f:
        dataset_dict = json.load(f)
        
    train_indices = []
    valid_indices = []
    test_indices = []
    
    train_ratio = 0.9
    valid_ratio = 0.05
    new_dataset_dict = {}
    sampling = [500, 5000, 5000, 5000, 5000, 5000, 1000]
    ratios = [0.8, 0.1, 0.1]
        
    for case in case_dict.keys():
        if 'last_index' in case: continue
        case_indices = case_dict[case]
        old_case_indices = old_case_dict[case]
        new_indices = list(set(case_indices) - set(old_case_indices))
        train_set, valid_set, test_set = random_split(new_indices, train_ratio, valid_ratio)
        print(case, len(train_set), len(valid_set), len(test_set))
        dataset_dict['train'] += train_set
        dataset_dict['valid'] += valid_set
        dataset_dict['test'] += test_set
        
        
    for k, v in dataset_dict.items():
        print(k, len(v))
    write_to_json(train_indices, valid_indices, test_indices, data=dataset_dict,
                  path = "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/dataset_split_256_by_class_new.json")