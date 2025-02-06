import os
import json

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    return 

if __name__=='__main__':
    root_folder = '/home/user01/aiotlab/nmduong/BoneTumor/RAW/REAL_WSIs/phase2_training_data'
    out_path = '/home/user01/aiotlab/nmduong/BoneTumor/RAW/REAL_WSIs/_DENOISE2/metadata.json'
    
    all_metadatas = []
    cnt = 0
    for case in os.listdir(root_folder):
        if "Case" not in case: continue
        print("Process:", case)
        case_folder = os.path.join(root_folder, case)
        metadata_path = os.path.join(case_folder, 'metadata.json')
        
        metadata = read_json(metadata_path)
        for item_infor in metadata:
            item_infor['case'] = case
            all_metadatas.append(item_infor)
            cnt += 1
    
    write_json(all_metadatas, out_path)
    
    print(f"TOTAL - {cnt} images")