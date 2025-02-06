'''
Simply divide a dict (case/class)_dict into dataset_dict, divide equally on each criteria 
(by class if using class_dict and by case if case_dict)
NOTICE: Only use a subset of data to test or finetune
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

def write_to_json(data, path='./dataset_split.json'):
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
    
    train_ratio = 0.33
    valid_ratio = 0.33
    new_dataset_dict = {}
    sampling = [500, 5000, 5000, 5000, 5000, 5000, 1000]
    ratios = [0.8, 0.1, 0.1]
        
    for case in case_dict.keys():
        if 'last_index' in case: continue
        # if case in ["1", "2", "3", "9"]:
        #     print(f"skip {case}")
        #     continue
        case_indices = case_dict[case]
        old_case_indices = old_case_dict[case]
        # print(len(case_indices))
        
        new_indices = list(set(case_indices) - set(old_case_indices))   # added indices of class [case]
        indices_each_set = random_split(new_indices, train_ratio, valid_ratio)  # divide into different group of class [case]s
        # print(case, len(new_indices))
        
        for i, mode in enumerate(['train', 'valid', 'test']):
            # get the indices of each class in each set (in old)
            class_in_set_indices = set(old_case_indices).intersection(dataset_dict[mode])
            new_class_indices = list(class_in_set_indices) + indices_each_set[i]    # all class indices in mode (train, valid, test)
            random.shuffle(new_class_indices)
            # print(case, len(new_class_indices))
            
            sampled_class_indices = new_class_indices[:min(int(sampling[int(case)]*ratios[i]), len(new_class_indices))]
            
            
            print(case, len(sampled_class_indices))
            new_dataset_dict[mode] = new_dataset_dict.get(mode, []) + sampled_class_indices
        
        # train_set, valid_set, test_set = random_split(new_indices, train_ratio, valid_ratio)
        # train_indices += train_set
        # valid_indices += valid_set
        # test_indices += test_set
    
    for k, v in new_dataset_dict.items():
        print(k, len(v))
    write_to_json(new_dataset_dict,
                  path = "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/dataset_split_256_by_class_sampled.json")