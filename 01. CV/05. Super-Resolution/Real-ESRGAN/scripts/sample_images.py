# 从hr_croped_time_region中抽取大约800张图片与psnr18multiscale混合作为训练集
import os

if __name__ == '__main__':
    images_dir = '/root/work/real-esrgan/train/datasets/landsea/raw_data/hr_croped_time_region'
    save_dir = '/root/work/real-esrgan/train/datasets/landsea/train/hr'
    import random
    import glob
    import shutil
    images_path = glob.glob(os.path.join(images_dir, '*.png'))
    for images_path in images_path:
        if random.random() < 0.05:
            shutil.copy2(images_path, save_dir)

