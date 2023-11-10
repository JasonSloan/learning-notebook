# 划分数据集（YOLO版本）
import os
import shutil
from random import random
from tqdm import tqdm


def train_val_split(src_images_root_dir, dst_images_root_dir_train, dst_images_root_dir_val, train_ratio):
    """划分训练集和验证集"""
    pbar = tqdm(os.listdir(src_images_root_dir), desc="Splitting dataset")
    for img_name in pbar:
        if random() < train_ratio:
            # 移动图片
            shutil.copy2(os.path.join(src_images_root_dir, img_name), dst_images_root_dir_train)
            # 移动标签
            label_name = img_name.split('.')[0]+'.txt'
            src_labels_root_dir = src_images_root_dir.replace("images", "labels")
            dst_labels_root_dir_train = dst_images_root_dir_train.replace("images", "labels")
            shutil.copy2(os.path.join(src_labels_root_dir, label_name), dst_labels_root_dir_train)
        else:
            # 移动图片
            shutil.copy2(os.path.join(src_images_root_dir, img_name), dst_images_root_dir_val)
            # 移动标签
            label_name = img_name.split('.')[0]+'.txt'
            src_labels_root_dir = src_images_root_dir.replace("images", "labels")
            dst_labels_root_dir_val = dst_images_root_dir_val.replace("images", "labels")
            shutil.copy2(os.path.join(src_labels_root_dir, label_name), dst_labels_root_dir_val)

def val_test_split(dst_images_root_dir_val, dst_images_root_dir_test):
    """划分验证集和测试集"""
    pbar = tqdm(os.listdir(dst_images_root_dir_val), desc="Splitting dataset")
    for img_name in pbar:
        if random() < 0.5:
            # 移动图片
            shutil.copy2(os.path.join(dst_images_root_dir_val, img_name), dst_images_root_dir_test)
            # 移动标签
            label_name = img_name.split('.')[0]+'.txt'
            dst_labels_root_dir_val = dst_images_root_dir_val.replace("images", "labels")
            dst_labels_root_dir_test = dst_images_root_dir_test.replace("images", "labels")
            shutil.copy2(os.path.join(dst_labels_root_dir_val, label_name), dst_labels_root_dir_test)

if __name__ == '__main__':
    src_images_root_dir = '/root/work/yolov5/datasets/wheelhouse_person_detection/raw_data/images'
    dst_images_root_dir_train = '/root/work/yolov5/datasets/wheelhouse_person_detection/images/train'
    dst_images_root_dir_val = '/root/work/yolov5/datasets/wheelhouse_person_detection/images/val'
    dst_images_root_dir_test = '/root/work/yolov5/datasets/wheelhouse_person_detection/images/test'
    train_val_split(src_images_root_dir,  dst_images_root_dir_train, dst_images_root_dir_val, train_ratio=0.8)
    val_test_split(dst_images_root_dir_val, dst_images_root_dir_test)