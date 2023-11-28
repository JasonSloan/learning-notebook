# 按照sole_images.txt（该文件中保存的是不重复的图像名字）中的图像名字，索引到对应的图像，然后将这些图像复制到hr_sole（lr_sole)文件夹中
# 与detect_duplicate_images.py脚本配合使用
import os
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    src_images_root_dir = '/root/work/real-esrgan/train/datasets/landsea/raw_data'
    src_images_sub_dirs = ['hr_sole_psnr18', 'lr_sole_psnr18']
    dst_images_root_dir = '/root/work/real-esrgan/train/datasets/landsea/raw_data'
    dst_images_sub_dirs = ['hr_sole_psnr15', 'lr_sole_psnr15']
    keep_file_path = '../datasets/landsea/raw_data/sole_images_psnr15.txt'
    for dst_sub_dir in dst_images_sub_dirs:
        os.makedirs(os.path.join(dst_images_root_dir, dst_sub_dir), exist_ok=True)
    for idx, src_sub_dir in enumerate(src_images_sub_dirs):
        print(f"Processing {src_sub_dir}")
        src_images_full_dir = os.path.join(src_images_root_dir, src_sub_dir)
        src_images = os.listdir(src_images_full_dir)
        with open(keep_file_path, 'r') as f:
            keep_images = f.readlines()
        keep_images = [x.strip() for x in keep_images]
        pbar = tqdm(src_images, total=len(src_images))
        count = 0
        for src_image in pbar:
            if src_image in keep_images:
                src_image_full_path = os.path.join(src_images_full_dir, src_image)
                dst_image_full_dir = os.path.join(dst_images_root_dir, dst_images_sub_dirs[idx])
                shutil.copy2(src_image_full_path, dst_image_full_dir)
                count += 1
        assert len(keep_images) == count