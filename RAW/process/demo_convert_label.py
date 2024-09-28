import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt



# Viable tumor - Green: RGB(0, 128, 0)
# Necrosis - Pink: RGB(255, 143, 204)
# Fibrosis/Hyalination - Red: RGB(255, 0, 0)
# Hemorrhage/Cystic change - Black: RGB(0, 0, 0)
# Inflammatory- Brown: RGB(165, 42, 42)
# Non-tumor tissue - Blue: RGB(0, 0, 255)

idx2color = [
    [255, 255, 255],
    [0, 128, 0],
    [255, 143, 204],
    [255, 0, 0],
    [0, 0, 0],
    [165, 42, 42],
    [0, 0, 255],]
# color2idx = {c: i for i, c in enumerate(idx2color)}

def apply_threshold_mapping(image, target_colors, tolerance):
    # Create masks for pixels that are closer to green or pink
    # Initialize the output image with the original image
    output = np.ones_like(image)*255  # 2D only
    masks = []
    for idx, color in enumerate(target_colors):
        color = np.array(color)
        mask = np.all(np.abs(image - color) < tolerance, axis=-1)
        output[mask] = color
        # output[mask] = idx

    return output

def open_img(path):
    out = cv2.imread(path)[:, :, :3]
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

label_folder = "/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/training_data/labels"
index = 42
label_path = os.path.join(label_folder, f"gt_{index}.png")
label_path = '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/phase2_training_data/Case_2/labels/slide-2024-07-26T08-00-29-R1-S2-labels-crop-1.png'

label = open_img(label_path)
fixed_label = apply_threshold_mapping(label, idx2color, 50)
plt.imsave('./fixed_sample.png', fixed_label)
print(fixed_label)