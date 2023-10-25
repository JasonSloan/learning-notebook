from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from pathlib import Path


def show_anns(anns):
    """SAM分割的可视化"""
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def save_labels(img, img_path, img_folder, masks, low_thread, min_area, max_area):
    """将SAM分割出来的框转换成yolo格式的label"""
    high_thread = 1 / low_thread
    masks_wh = np.array([mask['bbox'][2] / (mask['bbox'][3] + 0.001) for mask in masks])  # w/h
    masks_hw = np.array([mask['bbox'][3] / (mask['bbox'][2] + 0.001) for mask in masks])  # h/w
    masks_min_wh = np.min((masks_wh, masks_hw), axis=0)  # 选择最小的
    masks_max_wh = np.max((masks_wh, masks_hw), axis=0)  # 选择最大的
    # 选择长宽比例在low_thread-high_thread之间的，np.logical_and:逻辑与
    masks_mask = np.logical_and(masks_min_wh > low_thread, masks_max_wh < high_thread)
    # 选择面积在min_area-max_area之间的
    masks_area = np.array([mask['area'] for mask in masks])
    masks_mask = np.logical_and(masks_mask, masks_area > min_area)
    masks_mask = np.logical_and(masks_mask, masks_area < max_area)
    # 选择符合条件的mask
    masks = np.array(masks)[masks_mask]
    img_ = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W, _ = img.shape
    img_name = Path(img_path).stem
    label_path = os.path.join(img_folder, img_name + ".txt")
    for mask in masks:
        bbox = mask['bbox']
        x = (bbox[0] + bbox[2] / 2) / W  # x、y是中心点的坐标
        y = (bbox[1] + bbox[3] / 2) / H
        w = bbox[2] / W
        h = bbox[3] / H
        cords = " ".join([str(round(x, 6)) for x in [x, y, w, h]])
        cords = "0 " + cords  # 0 is the class id  TODO: add class id
        with open(label_path, "w", encoding="utf-8") as fp:
            fp.write(cords)

        # 下面这段是验证计算的label是否正确的代码，在原图上画出来，可以注释掉
        x1 = (x - w / 2) * W
        y1 = (y - h / 2) * H
        x2 = (x + w / 2) * W
        y2 = (y + h / 2) * H
        cv2.rectangle(img_, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    rectangle_path = os.path.join(img_folder, img_name + "_rectangle.png")
    cv2.imwrite(rectangle_path, img_)
    print(f"Save img to {rectangle_path}")
    # cv2.imshow('img', img_)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    sam_checkpoint = "./model/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device:{device}")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    print("Model loaded!")
    mask_generator = SamAutomaticMaskGenerator(sam)
    img_folder = "./data/"
    imgs_names = os.listdir(img_folder)
    for img_name in imgs_names:
        img_path_ = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path_)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start = time.time()
        masks = mask_generator.generate(img)
        end = time.time()
        print(f"Inference time: {round((end - start), 2)}s")
        # low_thread：SAM分割出来的框的达到要求的最小宽高比，min_area：SAM分割出来的框达到要求的最小面积，max_area：SAM分割出来的框达到要求的最大面积
        # save_labels(img, img_path, img_folder, masks, low_thread=0.65, min_area=5e3, max_area=9e4)  # 这几个参数可以自己调整

        # 下面这段是验证SAM分割的可视化代码
        plt.figure(figsize=(20, 20))
        plt.imshow(img)
        show_anns(masks)
        plt.axis('off')
        plt.savefig(f"{img_name}_sam.jpg")
        # plt.show()
