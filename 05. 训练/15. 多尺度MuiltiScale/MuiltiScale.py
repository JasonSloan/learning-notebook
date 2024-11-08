import random
import math
import torch.nn as nn
import torch


def muilti_scale(imgs, imgsz, gs):
    # imgs: 一个批次的图像数据， imgsz:原图尺寸， gs: 图片缩放后可以被整除的数
    # 缩放到原图的0.5-1.5倍之间，且缩放后的尺寸可以被gs整除
    # sz: 缩放后的长边尺寸， sf: 缩放比例
    sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
    sf = sz / max(imgs.shape[2:])  # scale factor
    if sf != 1:
        # 缩放后的最终新尺寸（长边缩放到sz的大小，短边按比例缩放后再变成可以被gs整除的大小）
        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
    return imgs, sf, ns


if __name__ == '__main__':
    imgs = torch.randn(16, 3, 640, 640)
    # imgs: 新的图像数据， sf: 长边缩放比例， ns: 缩放后的新尺寸
    imgs, sf, ns = muilti_scale(imgs, 640, 32)