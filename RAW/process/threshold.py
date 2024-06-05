import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
Image.MAX_IMAGE_PIXELS=None

def get_bbox(binary_img):
    h, w, c = binary_img.shape
    start_x, end_x, start_y, end_y = 0, 0, 0, 0
    
    done_h = False
    for i in range(h):
        row_sum = binary_img[i, ...].sum()
        if row_sum < 255*w*c:
            if not done_h:
                start_x = i-1
                done_h = True
            else:
                end_x = i+1
                
    done_w = False
    for i in range(w):
        col_sum = binary_img[:,i,:].sum()
        if col_sum < 255*h*c:
            if not done_w:
                start_y = i-1
                done_w = True
            else:
                end_y = i+1
                
    return start_x, end_x, start_y, end_y
    
def cut_img(img):
    binary_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w= binary_img.shape
    _, binary_img = cv2.threshold(binary_img, 255, 255, cv2.THRESH_BINARY)
    x1, x2, y1, y2 = get_bbox(img)
    print(x1, x2, y1, y2)
    cut_img = img[
        max(0, x1):min(x2, h), max(0, y1):min(y2, w),:]
    return cut_img

# ret, thresh = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)

def main():
    
    img_path = '/mnt/disk4/nmduong/Vin-Uni-Bone-Tumor/RAW/REAL_WSIs/slides/slide-2024-05-30T16-20-02-R1-S3-level2.png'
    name = os.path.basename(img_path)
    save_path = os.path.join(
        "/mnt/disk4/nmduong/Vin-Uni-Bone-Tumor/RAW/REAL_WSIs/slides/", 'cut_' + name[:-4] + '.png'
    )
    
    img = np.asarray(Image.open(img_path).convert("RGB"))
    img = (img[:, :, :3]).astype(np.uint8)
    
    binary_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Displaying the output image
    thresh = cut_img(img)
    Image.fromarray(thresh[:, :, :3]).save(save_path)
    
if __name__ == '__main__':
    main()