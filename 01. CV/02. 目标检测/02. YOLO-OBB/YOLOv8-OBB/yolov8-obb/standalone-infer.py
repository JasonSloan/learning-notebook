import math
import cv2
import torch
import numpy as np


class InferController:

    def __init__(self, img_size, weights, conf_thre, iou_thre) -> None:
        self.model = torch.load(weights, map_location="cpu")["model"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval().float().to(self.device)
        self.img_size = img_size
        self.conf_thre = conf_thre
        self.iou_thre = iou_thre

    @torch.no_grad
    def infer(self, img):
        img = self.preprocess(img)
        img = img.to(self.device).float()
        pred = self.model(img)[0].detach().cpu().float().transpose(-1, -2)
        result = self.postprocess(pred)
        return result

    def preprocess(self, img):
        img = self.letterbox(img)  # padded resize
        img = img / 255.0
        img = img[..., ::-1].transpose((2, 0, 1))[None]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous
        return torch.from_numpy(img)

    def letterbox(self, img):
        h0, w0 = img.shape[:2]
        h, w = self.img_size
        r = min(h / h0, w / w0)
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        w, h = math.ceil(new_unpad[0] / 32) * 32, math.ceil(new_unpad[1] / 32) * 32
        dw, dh = w - new_unpad[0], h - new_unpad[1]
        dw, dh = dw / 2, dh / 2
        left, right = round(dw - 0.1), round(dw + 0.1)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114]
        )
        self.im0_shape = (h0, w0)  # 给clip_boxes用
        self.ratio_pad = [r, [left, top, right, bottom]]
        return img

    def postprocess(self, pred):
        keep = self.non_max_suppression_roated(pred)
        keep[:, :5] = self.regularize_rboxes(keep[:, :5])
        keep[..., :4] = self.scale_boxes(keep[..., :4]).round()
        return keep

    def non_max_suppression_roated(self, pred, maxwh=7680):
        nc = pred.shape[-1] - 5  # DOTA1 dataset has 15 classes
        pred = pred[0, ...]  # 只支持batch_size=1
        boxes = torch.cat([pred[..., :4], pred[:, -1:]], dim=1)
        scores = pred[:, 4 : 4 + nc]
        classes = scores.argmax(1)
        scores = scores[list(range(scores.shape[0])), classes]
        smask = scores > self.conf_thre
        scores = scores[smask]
        classes = classes[smask]
        boxes = boxes[smask]
        sorted_idx = torch.argsort(scores, descending=True)
        scores = scores[sorted_idx][..., None]
        classes = classes[sorted_idx][..., None]
        boxes = boxes[sorted_idx]
        offset = classes * maxwh
        boxes_offset = torch.cat([boxes[:, :2]+offset, boxes[:, 2:]], dim=1)
        if boxes.shape[0] == 0:
            return torch.empty([0, 7])
        ious = self.batch_probiou(boxes_offset, boxes_offset).triu_(
            diagonal=1
        )  # 计算所有boxes与boxes之间的iou; triu_: 将主轴下方的元素置为0
        pick = torch.nonzero(ious.max(dim=0)[0] < self.iou_thre).squeeze_(-1)
        boxes = boxes[sorted_idx[pick]]
        scores = scores[sorted_idx[pick]]
        classes = classes[sorted_idx[pick]]
        return torch.cat([boxes, scores, classes], dim=1)   # xywhr+score+class

    def batch_probiou(self, obb1, obb2, eps=1e-7):
        """
        Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

        Args:
            obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
            obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
        """
        obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
        obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

        x1, y1 = obb1[..., :2].split(1, dim=-1)  # x1,y1: [N, 1]
        x2, y2 = (
            x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1)
        )  # x2,y2: [1, N]
        a1, b1, c1 = self._get_covariance_matrix(obb1)  # a1, b1, c1: [N, 1], 对应于公式(1)
        a2, b2, c2 = (
            x.squeeze(-1)[None] for x in self._get_covariance_matrix(obb2)
        )  # a2, b2, c2: [1, N], 对应于公式(1)

        t1 = (
            ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
            / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        ) * 0.25  # t1: [N, N];   t1 + t2为公式(8)
        t2 = (
            ((c1 + c2) * (x2 - x1) * (y1 - y2))
            / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        ) * 0.5  # t2: [N, N]    t1 + t2为公式(8)
        t3 = (
            ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
            / (
                4
                * (
                    (a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)
                ).sqrt()
                + eps
            )
            + eps
        ).log() * 0.5  # t3: [N, N]    t3为公式(9)
        bd = (t1 + t2 + t3).clamp(eps, 100.0)  # bd: [N, N]
        hd = (1.0 - (-bd).exp() + eps).sqrt()  # hd: [N, N]
        return 1 - hd

    def _get_covariance_matrix(self, boxes):
        """
        Generating covariance matrix from obbs.

        Args:
            boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

        Returns:
            (torch.Tensor): Covariance metrixs corresponding to original rotated bounding boxes.
        """
        # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.(论文中的公式15)
        gbbs = torch.cat(
            (boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1
        )  # 将OBB转换成GBB(guassion bounding  box):[N, 3]
        a, b, c = gbbs.split(1, dim=-1)  # a: W^2/12;  b: H^2/12;  c: theta
        cos = c.cos()
        sin = c.sin()
        cos2 = cos.pow(2)
        sin2 = sin.pow(2)
        return (
            a * cos2 + b * sin2,
            a * sin2 + b * cos2,
            (a - b) * cos * sin,
        )  # 对应于公式(1)

    def regularize_rboxes(self, rboxes):
        """
        Regularize rotated boxes in range [0, pi/2].

        Args:
            rboxes (torch.Tensor): (N, 5), xywhr.

        Returns:
            (torch.Tensor): The regularized boxes.
        """
        x, y, w, h, t = rboxes.unbind(dim=-1)
        # Swap edge and angle if h >= w
        w_ = torch.where(w > h, w, h)
        h_ = torch.where(w > h, h, w)
        t = torch.where(w > h, t, t + math.pi / 2) % math.pi
        return torch.stack([x, y, w_, h_, t], dim=-1)  # regularized boxes

    def scale_boxes(self, boxes):
        gain = self.ratio_pad[0]
        pad = self.ratio_pad[1]

        boxes[:, 0] -= pad[0]  # x1 padding
        boxes[:, 1] -= pad[1]  # y1 padding
        boxes /= gain
        boxes = self.clip_boxes(boxes)
        return boxes

    def clip_boxes(self, boxes):
        """
        Args: boxes: ndarray, shape=[n, 4], 4: x1y1x2y2
        Return: boxes: ndarray, shape=[n, 4], 4: x1y1x2y2
        """
        shape = self.im0_shape  # shape: [h0, w0]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
        return boxes


def xywhr2xyxyxyxy(center):
    # reference: https://github.com/ultralytics/ultralytics/blob/v8.1.0/ultralytics/utils/ops.py#L545
    is_numpy = isinstance(center, np.ndarray)
    cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)
    ctr = center[..., :2]
    w, h, angle = (center[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1) if is_numpy else torch.cat(vec1, dim=-1)
    vec2 = np.concatenate(vec2, axis=-1) if is_numpy else torch.cat(vec2, dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return (
        np.stack([pt1, pt2, pt3, pt4], axis=-2)
        if is_numpy
        else torch.stack([pt1, pt2, pt3, pt4], dim=-2)
    )


if __name__ == "__main__":
    image_path = "images/P0006.jpg"
    weights = "weights/yolov8s-obb.pt"
    conf_thre = 0.25
    iou_thre = 0.45
    image = cv2.imread(image_path)
    img_size = (640, 640)
    infercontroller = InferController(
        img_size=img_size, weights=weights, conf_thre=conf_thre, iou_thre=iou_thre
    )
    results = infercontroller.infer(image)
    for result in results:
        box = result[:4].detach().cpu().int().numpy().tolist()
        angle = result[4].detach().cpu().float().numpy().tolist()
        conf = result[5].detach().cpu().float().numpy().tolist()
        cls = result[6].detach().cpu().float().numpy().tolist()
        pts = xywhr2xyxyxyxy(
            torch.cat([torch.tensor(box), torch.tensor([angle])], dim=-1)
        )
        cv2.polylines(image, [np.asarray(pts, dtype=int)], True, [0, 0, 255], 2)
    cv2.imwrite("result.jpg", image)
    print("Done!")
