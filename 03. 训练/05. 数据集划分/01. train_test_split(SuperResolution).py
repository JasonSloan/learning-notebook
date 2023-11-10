# 划分数据集
import os
import shutil
from random import random
from tqdm import tqdm


def train_val_split(data_root_path, data_sub_dir, train_root_path, val_root_path, train_ratio):
    pbar = tqdm(os.listdir(os.path.join(data_root_path, data_sub_dir)), desc="Splitting dataset")
    for file in pbar:
        if random() < train_ratio:
            shutil.copy2(os.path.join(data_root_path, data_sub_dir, file), os.path.join(train_root_path, 'hr'))
            shutil.copy2(os.path.join(data_root_path, data_sub_dir.replace('hr', 'lr'), file), os.path.join(train_root_path, 'lr'))
        else:
            shutil.copy2(os.path.join(data_root_path, data_sub_dir, file), os.path.join(val_root_path, 'hr'))
            shutil.copy2(os.path.join(data_root_path, data_sub_dir.replace('hr', 'lr'), file), os.path.join(val_root_path, 'lr'))

def val_test_split(val_root_path, test_root_path):
    pbar = tqdm(os.listdir(os.path.join(val_root_path, 'hr')), desc="Splitting dataset")
    for file in pbar:
        if random() < 0.5:
            shutil.copy2(os.path.join(val_root_path, 'hr', file), os.path.join(test_root_path, 'hr'))
            shutil.copy2(os.path.join(val_root_path, 'lr', file), os.path.join(test_root_path, 'lr'))

if __name__ == '__main__':
    data_root_dir = '/root/work/real-esrgan/train/datasets/landsea/raw_data'
    data_sub_dir = 'faces'
    train_root_dir = '/root/work/real-esrgan/train/datasets/landsea/train'
    val_root_dir = '/root/work/real-esrgan/train/datasets/landsea/val'
    test_root_dir = '/root/work/real-esrgan/train/datasets/landsea/test'
    train_val_split(data_root_dir, data_sub_dir, train_root_dir, val_root_dir, train_ratio=0.8)
    # val_test_split(val_root_dir, test_root_dir)