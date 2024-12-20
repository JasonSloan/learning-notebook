# 本脚本实现, 将语义分割标注转换成yolov8-obb标注并保存

import os
import os.path as osp
from collections import namedtuple
import cv2
import numpy as np

Box = namedtuple('Box', ['class_index', 'box_points', 'theta', 'imgsz'])

def polygans2boxes(imgs_dir, labels_dir, show_box=False):
    """
    convert semantic segmentation polygans to minimum bounding rectangle, then transform to Box type.
    the orginal txt label stored in the format of 'class x1 y1 x2 y2 x3 y3......' 
    """
    imgs_names = os.listdir(imgs_dir)
    labels_names = [img_name.replace('images', 'labels').replace('jpg', 'txt') for img_name in imgs_names]
    new_labels_cache = {}
    for img_name, label_name in zip(imgs_names, labels_names):
        img_path = osp.join(imgs_dir, img_name)
        label_path = osp.join(labels_dir, label_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        with open(label_path) as fp:
            labels = fp.readlines()
        for label in labels:
            class_index = label.split(' ')[0]
            label = label.split(' ')[1:]
            polygons = [float(i) for i in label]
            polygons = np.asarray(polygons).reshape(-1, 2).astype(np.float32)
            rect = cv2.minAreaRect(polygons)
            box_points = cv2.boxPoints(rect).astype(np.float32)
            draw_pts = (box_points * np.array([w, h])).astype(np.int32)
            cv2.drawContours(img, [draw_pts], 0, (0, 255, 0), 2)  # Green color, thickness=2
            if show_box:
                while True:
                    cv2.imshow('Image with Min Area Rect', img)
                    key = cv2.waitKey(0)
                    if key:
                        break
            if label_name not in new_labels_cache.keys():
                new_labels_cache[label_name] = []
            box_one = Box(
                class_index=class_index, 
                box_points=box_points, 
                theta=None, imgsz=(h, w)
                )
            new_labels_cache[label_name].append(box_one)
    cv2.destroyAllWindows()
    
    return new_labels_cache
    

def boxes2yolotxt(labels_cache, save_dir):
    """
    convert the Box type data to yolo obb labels.
    yolo obb labels are stored in the format of 'class xc yc longside shortside theta'
    """
    os.makedirs(save_dir, exist_ok=True)
    for label_name in labels_cache:
        labels = []
        boxes = labels_cache[label_name]
        save_label_path = osp.join(save_dir, label_name)
        for box in boxes:
            box_points = box.box_points.ravel()
            label = str(box.class_index)
            for point in box_points:
                label = label + ' %g'%point
            labels.append(label)
            
        with open(save_label_path, 'w') as fp:
            fp.writelines(labels)
        labels.clear()

if __name__ == "__main__":
    imgs_dir = 'images'
    labels_dir = 'labels'
    new_labels_dir = 'new-labels'
    boxes = polygans2boxes(imgs_dir, labels_dir, show_box=False)
    boxes2yolotxt(boxes, new_labels_dir)
        
