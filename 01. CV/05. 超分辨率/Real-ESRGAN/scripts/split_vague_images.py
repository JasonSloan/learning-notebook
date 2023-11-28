# 按照vague_images.txt（该文件中的vague图像为手动挑选）中的图片索引，将图片分为vague_images和without_vague_images
import os
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    vague_images_file_path = '/root/work/data/vague_images.txt'
    with open(vague_images_file_path) as fp:
        vague_images_list = fp.read()
    vague_images_list = vague_images_list.split('\n')
    vague_images_indexes = ['{:06d}'.format(int(i)) for i in vague_images_list if i]
    target_images_path = '/root/work/data/lr_sole'
    vague_images_save_path = '/root/work/data/lr_vague_images'
    images_without_vague_path = '/root/work/data/lr_without_vague_images'
    os.makedirs(vague_images_save_path, exist_ok=True)
    os.makedirs(images_without_vague_path, exist_ok=True)
    target_images = os.listdir(target_images_path)
    pbar = tqdm(target_images)
    for target_image in pbar:
        if target_image.split('_')[0] in vague_images_indexes:
            shutil.copy2(os.path.join(target_images_path, target_image), vague_images_save_path)
        else:
            shutil.copy2(os.path.join(target_images_path, target_image), images_without_vague_path)




