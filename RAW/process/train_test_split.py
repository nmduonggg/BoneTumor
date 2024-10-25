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

def write_to_json(train, valid, test, path='./dataset_split.json'):
    data = {
        'train': train,
        'valid': valid,
        'test': test
    }
    with open(path, 'w') as f:
        json.dump(data, f)
    return

if __name__=='__main__':
    
    case_dict_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/case_dict_256.json'
    with open(case_dict_path, 'r') as f:
        case_dict = json.load(f)
        
    train_indices = []
    valid_indices = []
    test_indices = []
    
    train_ratio = 0.9
    valid_ratio = 0.05
        
    for case in case_dict.keys():
        if 'last_index' in case: continue
        case_indices = case_dict[case]
        train_set, valid_set, test_set = random_split(case_indices, train_ratio, valid_ratio)
        train_indices += train_set
        valid_indices += valid_set
        test_indices += test_set
    
    # print(len(train_indices), len(valid_indices), len(test_indices))
    write_to_json(train_indices, valid_indices, test_indices,
                  path = "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/dataset_split_256.json")