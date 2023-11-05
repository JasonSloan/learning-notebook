# 手动标注时间区域，然后根据标注的区域裁剪出对应的时间区域（因为该模型在文字显示上比较差）
import random
import os
import cv2
import math
from tqdm import tqdm


def generate_channelNum_imagePath_map_dict(path):
    image_names = os.listdir(path)
    image_paths = [os.path.join(path, image_name) for image_name in image_names]
    channelNum_imagePath_map_dict = {}
    for image_path in image_paths:
        channelNum = image_path.split('/')[-1].split('_')[2]
        if channelNum not in channelNum_imagePath_map_dict.keys():
            channelNum_imagePath_map_dict[channelNum] = []
        else:
            channelNum_imagePath_map_dict[channelNum].append(image_path)
    return channelNum_imagePath_map_dict


if __name__ == '__main__':
    # x1y1x2y2 format，列表中第一个字典是左上角，第二个字典是右下角
    target_regions = {'ch01': [{'min': [0, 0, 908, 84], 'max': [58, 33, 1100, 180]},
                               {'min': [1080, 700, 1600, 865], 'max': [1228, 793, 1712, 960]}],
                      'ch02': [{'min': [0, 0, 903, 84], 'max': [55, 32, 1063, 231]},
                               {'min': [1167, 793, 1596, 867], 'max': [1237, 798, 1712, 960]}],
                      'ch03': [{'min': [0, 0, 696, 36], 'max': [32, 0, 867, 147]},
                               {'min': [1075, 696, 1509, 847], 'max': [1233, 791, 1712, 960]}],
                      'ch04': [{'min': [0, 0, 460, 78], 'max': [24, 52, 565, 179]},
                               {'min': [1177, 798, 1402, 886], 'max': [1237, 848, 1712, 960]}],
                      'ch05': [{'min': [0, 0, 459, 77], 'max': [25, 52, 569, 182]},
                               {'min': [1104, 778, 1401, 886], 'max': [1238, 850, 1712, 960]}],
                      'ch06': [{'min': [0, 0, 459, 81], 'max': [24, 51, 570, 153]},
                               {'min': [1145, 772, 1403, 887], 'max': [1237, 847, 1712, 960]}],
                      'ch07': [{'min': [0, 0, 464, 79], 'max': [20, 49, 516, 159]},
                               {'min': [1172, 804, 1402, 885], 'max': [1235, 849, 1712, 960]}],
                      'ch08': [{'min': [0, 0, 460, 80], 'max': [19, 50, 539, 160]},
                               {'min': [1161, 793, 1431, 888], 'max': [1236, 849, 1712, 960]}],
                      'ch09': [{'min': [0, 0, 460, 80], 'max': [19, 50, 539, 160]},
                               {'min': [1161, 793, 1431, 888], 'max': [1236, 849, 1712, 960]}],
                      'ch10': [{'min': [0, 0, 460, 77], 'max': [25, 53, 574, 184]},
                               {'min': [1167, 793, 1403, 885], 'max': [1233, 847, 1712, 960]}],
                      'ch11': [{'min': [0, 0, 460, 77], 'max': [25, 53, 574, 184]},
                               {'min': [1167, 793, 1433, 887], 'max': [1235, 847, 1712, 960]}],
                      'ch12': [{'min': [0, 0, 460, 80], 'max': [25, 53, 574, 184]},
                               {'min': [1167, 793, 1331, 889], 'max': [1238, 849, 1712, 960]}],
                      'ch13': [{'min': [0, 0, 460, 80], 'max': [25, 53, 574, 184]},
                               {'min': [1167, 793, 1433, 887], 'max': [1238, 849, 1712, 960]}],
                      'ch14': [{'min': [0, 0, 460, 80], 'max': [25, 53, 574, 184]},
                               {'min': [1167, 793, 1331, 886], 'max': [1238, 849, 1712, 960]}],
                      'ch15': [{'min': [0, 0, 690, 60], 'max': [34, 21, 835, 177]},
                               {'min': [None, None, None, None], 'max': [None, None, None, None]}],
                      'ch16': [{'min': [0, 0, 460, 80], 'max': [25, 53, 574, 184]},
                               {'min': [1167, 793, 1304, 885], 'max': [1234, 848, 1712, 960]}],
                      'ch17': [{'min': [0, 0, 690, 60], 'max': [34, 21, 835, 177]},
                               {'min': [1080, 700, 1372, 841], 'max': [1228, 793, 1712, 960]}],
                      'ch18': [{'min': [0, 0, 690, 87], 'max': [37, 49, 1100, 180]},
                               {'min': [1080, 700, 1374, 890], 'max': [1236, 854, 1712, 960]}],
                      'ch19': [{'min': [0, 0, 690, 87], 'max': [37, 49, 1100, 180]},
                               {'min': [1080, 700, 1460, 899], 'max': [1231, 849, 1712, 960]}],
                      'ch20': [{'min': [0, 0, 690, 87], 'max': [37, 49, 1100, 180]},
                               {'min': [1080, 700, 1411, 897], 'max': [1236, 854, 1712, 960]}],
                      'ch21': [{'min': [0, 0, 690, 87], 'max': [37, 49, 1100, 180]},
                               {'min': [1080, 700, 1416, 899], 'max': [1236, 854, 1712, 960]}],
                      'ch22': [{'min': [0, 0, 486, 87], 'max': [23, 52, 1100, 180]},
                               {'min': [1080, 700, 1350, 880], 'max': [1236, 854, 1712, 960]}],
                      'ch23': [{'min': [0, 0, 904, 87], 'max': [60, 32, 1200, 180]},
                               {'min': [1080, 700, 1568, 863], 'max': [1239, 801, 1712, 960]}]
                      }
    channelNum_imagePath_map_dict = generate_channelNum_imagePath_map_dict('../datasets/landsea/raw_data/hr/')
    for channel_num in target_regions.keys():
        print("=" * 50 + f"channel_num: {channel_num}" + "=" * 50)
        region_cords = target_regions[channel_num]
        images_path = channelNum_imagePath_map_dict[channel_num]
        pbar = tqdm(images_path, total=len(images_path))
        for image_path in pbar:
            org_img_hr = cv2.imread(image_path)
            org_img_lr = cv2.imread(image_path.replace('hr', 'lr'))
            for idx, region_cord in enumerate(region_cords):
                if region_cord['min'] == [None, None, None, None]:
                    continue
                x1 = random.randint(region_cord['min'][0], region_cord['max'][0])
                y1 = random.randint(region_cord['min'][1], region_cord['max'][1])
                x2 = random.randint(region_cord['min'][2], region_cord['max'][2])
                y2 = random.randint(region_cord['min'][3], region_cord['max'][3])
                # x2 -x1 以及 y2 -y1 必须是4的倍数
                x2 = x2 - (x2 - x1) % 4
                y2 = y2 - (y2 - y1) % 4
                croped_image_hr = org_img_hr[y1:y2, x1:x2]
                croped_image_save_path = image_path.replace('hr', 'hr_croped_time_region').replace('.png', f'_{idx}.png')
                cv2.imwrite(croped_image_save_path, croped_image_hr)
                x1_ = math.floor(x1 / 4)
                y1_ = math.floor(y1 / 4)
                x2_ = math.floor(x2 / 4)
                y2_ = math.floor(y2 / 4)
                assert (x2_ - x1_) * 4 == (x2 - x1)
                assert (y2_ - y1_) * 4 == (y2 - y1)
                croped_image_lr = org_img_lr[y1_:y2_, x1_:x2_]
                croped_image_save_path = image_path.replace('hr', 'lr_croped_time_region').replace('.png',
                                                                                                   f'_{idx}.png')
                cv2.imwrite(croped_image_save_path, croped_image_lr)
