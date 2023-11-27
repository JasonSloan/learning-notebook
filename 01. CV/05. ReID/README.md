# 一. ReID问题



# 二. ReID评估指标

## 1. Rank-1

表示在候选库中得到与检索目标相似度排名最高的图片为目标行人的概率。

## 2. **mAP**

precision: 正确被检测的(TP)占所有实际被检测(TP+FP)的比例

precision = 查询返回的正确个数 / 返回的总个数 

recall: 正确被检测的(TP)占所有应该被检测(TP+FN)的比例
recall = 查询返回的正确个数 / 该查询对象在gallery中的总个数

AP为PR曲线的下面积; mAP为AP求平均

## 3. CMC

