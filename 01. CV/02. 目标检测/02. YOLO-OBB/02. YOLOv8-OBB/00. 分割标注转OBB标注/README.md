## 一. yolov5-converter

[yolov5-obb训练代码](https://github.com/hukaixuan19970627/yolov5_obb)

[作者写的讲解博客](https://zhuanlan.zhihu.com/p/358072483)

根据训练代码的要求, 训练用的标注数据需满足[长边表示法](https://zhuanlan.zhihu.com/p/459018810), 即下图中的格式; 但是标注数据一般为语义分割的polygans点集, 因此yolov5-converter.py脚本实现了将语义分割的polygans点集转换为该训练代码需求的格式

长边表示法:

![](assets/1.jpg)

## 二.  yolov8-converter

[yolov8-obb文档](https://docs.ultralytics.com/tasks/obb/)

yolov8中训练用的标注数据需要标注好满足旋转框的四个顶点坐标即可, 也就是class x1 y1 x2 y2 x3 y3 x4 y4。

但是标注数据一般为语义分割的polygans点集, 因此yolov8-converter.py脚本实现了将语义分割的polygans点集转换为该训练代码需求的格式





