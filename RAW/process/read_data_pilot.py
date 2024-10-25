import os
import numpy as np
import matplotlib.pyplot as plt
import json

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

def open_img(img_path):
    out = cv2.imread(img_path)[:, :, :3]
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out

def cut_image(image,  
                crop_sz=224, step=200,
                outdir = "./RAW/REAL_WSIs/training_data/", gt=False):
    global im_index, gt_index
    
    image_outdir = os.path.join(outdir, "images")
    label_outdir = os.path.join(outdir, "labels")
    # print("Create folder: ", image_outdir, label_outdir)
    os.makedirs(image_outdir, exist_ok=True)
    os.makedirs(label_outdir, exist_ok=True)
    
    h, w, c = image.shape
    # cut WSIs into tiles
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    num_h = 0
    
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            crop_img = image[x:x + crop_sz, y:y + crop_sz, :]
            print([x, x+crop_sz, y, y+crop_sz], end= ' ')
            # crop_lbl = label[x:x + crop_sz, y:y + crop_sz, :]
            
            # # Create the mask
            # mask = np.all(crop_lbl == [255, 255, 255], axis=-1)
            # # Convert the boolean mask to an integer mask (0 for white, 1 for other colors)
            # mask = np.expand_dims(np.where(mask, 0, 1), axis=2).repeat(3, axis=-1)
            # crop_img = crop_img * mask + np.ones_like(crop_img)*255 * (1-mask)
            
            # img_mask = np.all(image == [255, 255, 255], axis=-1)
            # img_mask = np.expand_dims(np.where(img_mask, 0, 1), axis=2).repeat(3, axis=-1)
            # label = label * img_mask + np.ones_like(label)*255 * (1-img_mask)
            
            # if np.mean(crop_img)==255:
            #     continue
            
            if not gt:
                plt.imsave(os.path.join(image_outdir, f"patch_{im_index}.png"), crop_img.astype(np.uint8))
                im_index += 1
            else:
                plt.imsave(os.path.join(label_outdir, f"gt_{gt_index}.png"), crop_img.astype(np.uint8))
                gt_index += 1
            
    h=x + crop_sz
    w=y + crop_sz
    # return lr_list, num_h, num_w, h, w
    
    # print(f"Done extract {index} tiles")
            
            


if __name__ == "__main__":
    cases = [1, 2, 4]
    image_folder = "./RAW/REAL_WSIs/images/"
    label_folder = "./RAW/REAL_WSIs/labels/"
    # case_dict_outpath = "./RAW/REAL_WSIs/case_dict.json"
    label_name = "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/labels/Case_1/slide-2024-07-25T13-21-35-R1-S1-x8-labels.jpg"
    img_name = "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/images/Case_1/slide-2024-07-25T13-21-35-R1-S1-resize.png"
    
    case_dict = {}  # keep track which patches belong to which case
    
    # Configuration
    crop_sz = 3000
    step = 3000
    im_index = 0
    gt_index = 0
    
    # Training and Valid
    upsample=False
    label = open_img(label_name)
    
    if "x8" in label_name:
        # img_name = label_name.split("-x8")[0] + '.png'
        upsample = True
    # else:
    #     img_name = label_name.split("-labels")[0] + '.png'
        
    image_path = os.path.join(img_name)
    image = open_img(image_path)
    
    label = cv2.resize(label, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)
    
    # mask = np.all(label == [255, 255, 255], axis=-1)
    # # Convert the boolean mask to an integer mask (0 for white, 1 for other colors)
    # mask = np.expand_dims(np.where(mask, 0, 1), axis=2).repeat(3, axis=-1)
    # image = image * mask + np.ones_like(image)*255 * (1-mask)
    
    # img_mask = np.all(image == [255, 255, 255], axis=-1)
    # img_mask = np.expand_dims(np.where(img_mask, 0, 1), axis=2).repeat(3, axis=-1)
    # label = label * img_mask + np.ones_like(label)*255 * (1-img_mask)
    plt.imsave("./sample.png", image)
    plt.imsave("./sample_label.png", label)
    
    print(label.shape, image.shape)
    
    old_index = im_index
    cut_image(image ,
                    crop_sz=crop_sz, step=step,
                    outdir = "./RAW/REAL_WSIs/training_data/",
                    gt=False)
    print("Done label")
    cut_image(label , 
                    crop_sz=crop_sz, step=step,
                    outdir = "./RAW/REAL_WSIs/training_data/",
                    gt=True)
    # case_dict[tcase] = case_dict.get(tcase, []) + list(range(old_index, index))
    assert im_index==gt_index
    print(f"{old_index} to {im_index}")
            
    print(f"[Summary]: {im_index+1} patches in total")
    
    
# gt_8052.png