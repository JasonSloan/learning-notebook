## 目标检测中的评估指标

## 一. AP(以下指标假设都只针对一个类别)

TP:  与gt之间IoU大于IoU_threshold的bounding_box

FP:  与gt之间IoU小于IoU_threshold的bounding_box(包括完全无交集的)

FN: 漏检的gt框



