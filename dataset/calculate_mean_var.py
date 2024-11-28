import os
import shutil
import random
import json

import numpy as np
from PIL import Image

nuaa_dir = '/mnt/e/Dataset/NUAA-SIRST'
nudt_dir = '/mnt/e/Dataset/NUDT-SIRST'
irstd_dir = '/mnt/e/Dataset/IRSTD-1k'
target_dir = '/mnt/e/Dataset/DIL_SIRSTD'

nuaa_image_dir = os.path.join(nuaa_dir, 'images/images')
nuaa_mask_dir = os.path.join(nuaa_dir, 'masks/masks')
nudt_image_dir = os.path.join(nudt_dir, 'images')
nudt_mask_dir = os.path.join(nudt_dir, 'masks')
irstd_image_dir = os.path.join(irstd_dir, 'IRSTD1k_Img')
irstd_mask_dir = os.path.join(irstd_dir, 'IRSTD1k_Label')
target_image_dir = os.path.join(target_dir, 'images')
target_mask_dir = os.path.join(target_dir, 'masks')

num_image = 0
num_pixel = 0
mean = np.zeros(1)
var = np.zeros(1)
mean_list = []
variance_list = []
std_list = []
for i in range(7):
    image_dir = os.path.join(target_image_dir, str(i))
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.png'):
                image_file = os.path.join(root, file)
                image = Image.open(image_file).convert('I')
                image_np = np.array(image)/255.0
                num_image += 1
                mean += np.mean(image_np)
                var += np.var(image_np, ddof=1)
    mean_list.append((mean/num_image).item())
    variance_list.append((var/num_image).item())
    std_list.append(variance_list[i]**0.5)

print(mean_list)
print(std_list)