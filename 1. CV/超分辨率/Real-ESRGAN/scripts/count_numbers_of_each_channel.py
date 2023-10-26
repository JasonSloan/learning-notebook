# 按照通道ID统计每个通道的图片数量
import os


def count_numbers_of_each_channel(images_path):
    images_names = os.listdir(images_path)
    keys = [f"ch{i:02d}" for i in range(1, 24)]
    images_counts = {key: 0 for key in keys}
    for image_name in images_names:
        image_channel = image_name.split('_')[0]
        images_counts[image_channel] += 1
    return images_counts

if __name__ == "__main__":
    images_path = '/root/work/real-esrgan/train/datasets/landsea/raw_data/lr_sole_psnr15'
    images_counts = count_numbers_of_each_channel(images_path)
    print(images_counts)

