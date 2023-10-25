import os

import skimage
from tqdm import tqdm

def calculate_psnr(image1_path, image2_path):
    # Ensure both images have the same data type (typically uint8 for images)
    image1 = skimage.io.imread(image1_path)
    image1 = skimage.img_as_ubyte(image1)
    image2 = skimage.io.imread(image2_path)
    image2 = skimage.img_as_ubyte(image2)

    # Calculate the PSNR between the two images
    psnr = skimage.metrics.peak_signal_noise_ratio(image1, image2)
    return psnr


if __name__ == '__main__':
    # images_root_dir = '/root/work/real-esrgan/train/datasets/DIV2K_sub'
    # hr_image_names = os.listdir(images_root_dir + '/hr')
    # hr_images_path = [os.path.join(images_root_dir, 'hr', hr_image_name) for hr_image_name in hr_image_names]
    # psnr_all = 0
    # psnr_list = []
    # count = 0
    # pbar = tqdm(hr_images_path, desc="Calculating......")
    # for hr_image_path in pbar:
    #     fake_hr_image_path = hr_image_path.replace('hr', 'fake_hr').split('.')[0] + 'x4.png'
    #     psnr = calculate_psnr(hr_image_path, fake_hr_image_path)
    #     count += 1
    #     psnr_all += psnr
    #     psnr_list.append(psnr)
    # print('average psnr is: ', psnr_all / count)
    # print('psnr list is: ', psnr_list)
    path1 = '/root/work/data/hr/000183_715_ch12_20230715094225.png'
    path2 = '/root/work/data/hr/000183_721_ch12_20230721114208.png'
    print(calculate_psnr(path1, path2))

