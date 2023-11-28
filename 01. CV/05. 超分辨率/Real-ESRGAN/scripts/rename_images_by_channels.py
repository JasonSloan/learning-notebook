# 将通道名字放在前面，方便观察按照通道有多少重复图片

import os
from tqdm import tqdm
import shutil

if __name__ == "__main__":
    images_path = '/root/work/data/lr_without_vague_images'
    images_names = os.listdir(images_path)
    new_images_names = {}
    for image_name in images_names:
        new_images_names[image_name] = image_name.split('_')[2] + '_' + image_name
    new_images_dir = '/root/work/data/lr_without_vague_images_new'
    os.makedirs(new_images_dir, exist_ok=True)
    pbar = tqdm(new_images_names.items())
    for key, value in pbar:
        shutil.copy2(os.path.join(images_path, key), os.path.join(new_images_dir, value))
