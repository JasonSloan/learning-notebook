import warnings
warnings.filterwarnings("ignore")

import cv2
import torch
import torchvision
import numpy as np 

from model import Model       


class InferenceController:
    
    def __init__(self, cfg, weight, conf_thre=0.29, iou_thre=0.31, img_size=(384, 640)):
        model = Model(cfg)
        model_state_dict = torch.load(weight, map_location="cpu")
        model.load_state_dict(model_state_dict)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = self.device == "cuda"
        self.model = model.to(self.device).eval()
        if self.half:
            self.model = self.model.half()
        self.conf_thre = conf_thre
        self.iou_thre = iou_thre
        self.img_size = img_size

    @torch.no_grad    
    def infer(self, img):
        img = self.preprocess(img)
        img = img.to(self.device).float()
        if self.half:
            img = img.half()
        pred = self.model(img)[0].detach().cpu().float()
        result = self.postprocess(pred)
        return result
    
    def preprocess(self, img):
        img = self.letterbox(img)  # padded resize
        img = img / 255.
        img = img.transpose((2, 0, 1))[::-1][None]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous
        return torch.from_numpy(img)
    
    def postprocess(self, pred):
        """
        return: List: [box1, box2, ...]
        """
        kept = self.non_max_suppression(pred)
        if kept.shape[0] == 0:
            return []
        kept[..., :4] = self.scale_boxes(kept[..., :4]).round()
        return kept
            
    def letterbox(self, img):
        h0, w0 = img.shape[:2]
        h, w = self.img_size
        r = min(h / h0, w / w0)
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        dw, dh = w - new_unpad[0], h - new_unpad[1]
        dw, dh = dw / 2, dh / 2
        left, right = round(dw - 0.1), round(dw + 0.1)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114])
        self.im0_shape = (h0, w0)               # 给clip_boxes用
        self.ratio_pad = [r, [left, top]]
        return img
    
    def non_max_suppression(self, pred):
        """
        args: pred: ndarray, shape=[bs, 25200, 6], 6: xywh, obj_conf, cls_conf
        return: ndarray, shape=[n, 4], 4: x1y1x2y2
        """
        kept = torch.empty([0, 6])
        # pred : [bs, 25200, 6], 6: xywh, obj_conf, cls_conf
        pred = pred[0, ...]                     # 只支持batch_size=1
        conf = pred[:, 4]                       # conf = obj_conf * cls_conf
        cmask = conf > self.conf_thre
        pred = pred[cmask, :]
        if pred.shape[0] == 0:
            return kept
        scores = pred[:, 4:5] * pred[..., 5:]
        classes = scores.argmax(1)
        scores = scores[range(scores.shape[0]), classes]
        smask = scores > self.conf_thre
        scores = scores[smask]
        classes = classes[smask]
        boxes = pred[..., :4][smask]
        boxes = self.xywh2xyxy(boxes)
        kept_indices = torchvision.ops.batched_nms(boxes, scores, classes, self.iou_thre)
        kept = torch.cat([boxes[kept_indices], scores[kept_indices][..., None], classes[kept_indices][..., None]], dim=1)

        return kept

    def scale_boxes(self, boxes):
        gain = self.ratio_pad[0]
        pad = self.ratio_pad[1]

        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes /= gain
        boxes = self.clip_boxes(boxes)
        return boxes
    
    def clip_boxes(self, boxes):
        """
        Args: boxes: ndarray, shape=[n, 4], 4: x1y1x2y2
        Return: boxes: ndarray, shape=[n, 4], 4: x1y1x2y2
        """
        shape = self.im0_shape       # shape: [h0, w0]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
        return boxes
    
    def xywh2xyxy(self, boxes):
        xy = boxes[..., :2]
        wh = boxes[..., 2:]
        boxes[..., :2] = xy - wh / 2
        boxes[..., 2:] = xy + wh / 2
        return boxes
    
    def box_iou(self, box1, box2, eps=1e-7):
        """
        Args:
            box1 (_type_): ndarray, shape=[4], 4: x1y1x2y2
            box2 (_type_): ndarray, shape=[4], 4: x1y1x2y2
            eps (_type_, optional):  Defaults to 1e-7.

        Returns:
            float: iou
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = box1
        b2_x1, b2_y1, b2_x2, b2_y2 = box2
        inter_rect_x1 = max(b1_x1, b2_x1)
        inter_rect_y1 = max(b1_y1, b2_y1)
        inter_rect_x2 = min(b1_x2, b2_x2)
        inter_rect_y2 = min(b1_y2, b2_y2)
        inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * max(inter_rect_y2 - inter_rect_y1 + 1, 0)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / float(b1_area + b2_area - inter_area + eps)
        return iou
    
    def box_giou(self, box1, box2, eps=1e-7):
        """
        Args:
            box1 (_type_): ndarray, shape=[4], 4: x1y1x2y2
            box2 (_type_): ndarray, shape=[4], 4: x1y1x2y2
            eps (_type_, optional):  Defaults to 1e-7.

        Returns:
            float: giou
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = box1
        b2_x1, b2_y1, b2_x2, b2_y2 = box2
        inter_rect_x1 = max(b1_x1, b2_x1)
        inter_rect_y1 = max(b1_y1, b2_y1)
        inter_rect_x2 = min(b1_x2, b2_x2)
        inter_rect_y2 = min(b1_y2, b2_y2)
        enclosing_rect_x1 = min(b1_x1, b2_x1)
        enclosing_rect_y1 = min(b1_y1, b2_y1)
        enclosing_rect_x2 = max(b1_x2, b2_x2)
        enclosing_rect_y2 = max(b1_y2, b2_y2)
        inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * max(inter_rect_y2 - inter_rect_y1 + 1, 0)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        union_area = (b1_area + b2_area) - inter_area
        iou = inter_area / (union_area + eps)
        enclosing_area = (enclosing_rect_x2 - enclosing_rect_x1) * (enclosing_rect_y2 - enclosing_rect_y1)
        giou = iou - (enclosing_area - union_area) / (enclosing_area + eps)
        return giou
    

if __name__ == "__main__":
    import cv2
    import time 
    import glob
    import os

    cfg = 'yolov5-bytetrack/cfg/yolov5s.yaml'
    weight = "yolov5-bytetrack/weights/yolov5s.pt"

    controller = InferenceController(cfg, weight, conf_thre=0.01, iou_thre=0.3)
    # time_all = []
    # img_path = "yolov5-bytetrack/images/0002_1_000002.jpg"
    # img = cv2.imread(img_path)
    # for i in range(100):
    #     results = controller.infer(img)

    video_path = "videos/palace.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(int(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        controller.infer(frame)