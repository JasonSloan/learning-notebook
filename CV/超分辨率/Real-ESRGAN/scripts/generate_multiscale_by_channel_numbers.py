# 按照每个通道的图片数量决定是否扩充为多尺度
# 如果通道i的图片数量大于threashold，则从中随机选出threashold张
# 如果通道i的图片数量大于threashold/4，则从中随机选出threashold/4张，多尺度扩充到threashold张
# 如果通道i图片数量小于threashold，则多尺度扩充*4倍
import os
import math
import random
from PIL import Image
import shutil

from count_numbers_of_each_channel import count_numbers_of_each_channel


def generate_images_path_by_channel(images_path):
    images_path_by_channel = {}
    images_names = os.listdir(images_path)
    for image_name in images_names:
        image_channel = image_name.split('_')[0]
        if image_channel not in images_path_by_channel.keys():
            images_path_by_channel[image_channel] = []
        images_path_by_channel[image_channel].append(os.path.join(images_path, image_name))
    return images_path_by_channel


def generate_multiscale(image_path, save_path, imageName_size_mappings, is_hr_period, shortest_edge=400):
    # For DF2K, we consider the following three scales,
    # and the smallest image whose shortest edge is 400
    scale_list = [0.75, 0.5, 1 / 3]
    path_list = [image_path]
    for path in path_list:
        print(path)
        basename = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path)
        width, height = img.size
        img.save(os.path.join(save_path, f'{basename}T0.png'))
        make_exact_division = lambda x, scale: math.floor((x * scale) / 4) * 4
        for idx, scale in enumerate(scale_list):
            if is_hr_period:
                new_width = make_exact_division(width, scale)
                new_height = make_exact_division(height, scale)
                # 因为在多尺度过程中，存在hr图片resize后不能被4整除的情况，这样就不能形成pair对
                # 记录一下，如果是hr阶段那么就按照图像名字：图像尺寸记录；如果是lr阶段，那么就把hr阶段的记录拿出来按照resize图像
                lr_path = path.replace("hr", "lr")
                if lr_path not in imageName_size_mappings.keys():
                    imageName_size_mappings[lr_path] = []
                imageName_size_mappings[lr_path].append([int(new_width / 4), int(new_height / 4)])
            else:
                new_width = imageName_size_mappings[path][idx][0]
                new_height = imageName_size_mappings[path][idx][1]
            print(f'\t{scale:.2f}')
            rlt = img.resize((new_width, new_height), resample=Image.LANCZOS)
            rlt.save(os.path.join(save_path, f'{basename}T{idx + 1}.png'))

        # # # save the smallest image which the shortest edge is 400
        # if width < height:
        #     ratio = height / width
        #     width = shortest_edge
        #     height = int(width * ratio)
        # else:
        #     ratio = width / height
        #     height = shortest_edge
        #     width = int(height * ratio)
        # rlt = img.resize((int(width), int(height)), resample=Image.LANCZOS)
        # rlt.save(os.path.join(save_path, f'{basename}T{idx + 1}.png'))


def generate_mixed_datasets(images_path, save_path, threashold, shortest_edge):
    images_counts = count_numbers_of_each_channel(images_path)
    images_path_by_channel = generate_images_path_by_channel(images_path)
    channels_status = []
    for channel_id in images_counts.keys():
        images_number = images_counts[channel_id]
        if images_number == 0:
            continue
        if images_number > threashold:
            channels_status.append({"channel_id": channel_id, "multiscale": False,
                                    "images_path": random.sample(images_path_by_channel[channel_id], threashold)})
        elif images_number > (threashold / 4):
            channels_status.append({"channel_id": channel_id, "multiscale": True,
                                    "images_path": random.sample(images_path_by_channel[channel_id],
                                                                 int(threashold / 4))})
        else:
            channels_status.append(
                {"channel_id": channel_id, "multiscale": True, "images_path": images_path_by_channel[channel_id]})
    os.makedirs(save_path, exist_ok=True)
    imageName_size_mappings = {}
    for channel_status in channels_status:
        if channel_status["multiscale"]:
            for image_path in channel_status["images_path"]:
                generate_multiscale(image_path, save_path,
                                    imageName_size_mappings,
                                    is_hr_period=True,
                                    shortest_edge=400)
                generate_multiscale(image_path.replace("hr", "lr"),
                                    save_path.replace("hr", "lr"),
                                    imageName_size_mappings,
                                    is_hr_period=False,
                                    shortest_edge=100)
            else:
                for image_path in channel_status["images_path"]:
                    shutil.copy2(image_path, save_path)
                    shutil.copy2(image_path.replace("hr", "lr"), save_path.replace("hr", "lr"))


if __name__ == '__main__':
    images_path = '/root/work/real-esrgan/train/datasets/landsea/raw_data/hr_sole_psnr15'
    save_path = '/root/work/real-esrgan/train/datasets/landsea/raw_data/hr_sole_psnr15_mixed'
    threashold = 8
    # # for hr image
    # generate_mixed_datasets(images_path, save_path, threashold, shortest_edge=400)
    # for lr image
    generate_mixed_datasets(images_path, save_path, threashold, shortest_edge=400)
