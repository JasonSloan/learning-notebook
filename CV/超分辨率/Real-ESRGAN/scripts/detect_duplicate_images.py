# 检查一个文件夹内重复的图片，将不重复的图片的文件名保存到sole_images.txt中
# 与split_duplicate_images.py脚本配合使用
import os
import time
from tqdm import tqdm
import skimage.io
import skimage.metrics


def detect_duplicate_images(image_root_dir, psnr_threashold, save_dir):
    image_names = os.listdir(image_root_dir)
    image_names.sort()
    images_path = [os.path.join(image_root_dir, image_name) for image_name in image_names]
    keep = [True for i in range(len(images_path))]
    count = 0
    start = time.time()
    for i in range(len(images_path) - 1):
        print("=" * 50 + f"{image_names[i]}" + "=" * 50)
        if keep[i] is False:
            continue
        image1_channel_num = image_names[i].split('_')[0]
        image1 = skimage.io.imread(images_path[i])
        image1 = skimage.img_as_ubyte(image1)
        pbar = tqdm(range(i + 1, len(images_path)), ncols=150, leave=False)
        for j in pbar:
            if keep[j] is False:
                continue
            image2_channel_num = image_names[j].split('_')[0]
            if image1_channel_num != image2_channel_num:
                continue
            image2 = skimage.io.imread(images_path[j])
            image2 = skimage.img_as_ubyte(image2)
            psnr = skimage.metrics.peak_signal_noise_ratio(image1, image2)
            if psnr > psnr_threashold:
                keep[j] = False
                count += 1
                pbar.set_description(f'Image {image_names[i]} and image {image_names[j]} are similar!  psnr value is '
                                     f':{psnr:.2f}')
            else:
                pbar.set_description(f'Image {image_names[i]} and image {image_names[j]} psnr value is :{psnr:.2f}')
    print(f"{count} images are duplicated!")
    keep_images = [image_names[i] for i in range(len(keep)) if keep[i]]
    save_path = os.path.join(save_dir, f'sole_images_psnr{psnr_threashold}.txt')
    with open(save_path, 'w') as fp:
        for keep_image in keep_images:
            fp.write(keep_image + '\n')
    end = time.time()
    print(f"Total time: {end - start:.2f}s")

if __name__ == '__main__':
    image_root_dir = '/root/work/real-esrgan/train/datasets/landsea/raw_data/hr_sole_psnr18'
    psnr_threashold = 15
    save_dir = '/root/work/real-esrgan/train/datasets/landsea/raw_data'
    detect_duplicate_images(image_root_dir, psnr_threashold, save_dir)
