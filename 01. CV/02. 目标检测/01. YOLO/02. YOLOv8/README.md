# 一. YOLOv8改进对比实验

## 1. 来源

[CSDN迪菲赫尔曼专栏](https://yolov5.blog.csdn.net/article/details/130671688?spm=1001.2014.3001.5502)

## 2. 对比实验记录

试验数据：安全帽检测[链接](https://pan.baidu.com/s/16UU_PCMaHmMiEx53vfaJFQ ) ，提取码：81ck

训练集： 15887张； 验证集：4641张

数据统计： 面积小于 的目标？个

所有试验的训练轮数：200

试验控制变量1个，即当前试验更改的变量

variable type代表的是更换的变量的类型（更换骨干网络、激活函数、损失函数等）

测试的参数量以及浮点计算量均为float32类型的模型及数据，测试数据大小(1,3,640,640)

测试的模型规模均为s



以下试验对比为已经完成的以及近期内会继续完成的：

**先说结论：对于安全帽检测这个数据集，效果上有增益的改进由高到低：SPD-4head、SPD、SimAM、SE、SlimNeck、CA、iRMB、MPDIoU、ECA、AFPN、CARAFE；其中iRMB在速度上也有增益；SimAM、SE、MPDIoU、ECA在速度上没有增益；其余在速度上会有负增益。**

| index | model                                    | variable type | variable name                 | Params(Mb) | FLOPs(Gb) | precision | recall | mAP50  | mAP50-95 | FLOPs reudce  ratio | mAP50 gain ratio |
| ----- | ---------------------------------------- | ------------- | ----------------------------- | ---------- | --------- | --------- | ------ | ------ | -------- | ------------------- | ---------------- |
| 0     | yolov8s                                  | None          | None                          | 11.136     | 14.325    | 0.943     | 0.9205 | 0.9618 | 0.6157   | 0.00%               | 0.00%            |
| 1     | yolov8-MobileNext                        | backbone      | MobileNext                    | 7.249      | 9.358     | 0.9327    | 0.896  | 0.9483 | 0.5895   | 34.67%              | -1.40%           |
| 2     | yolov8-PP-LCNet                          | backbone      | PP-LCNet                      | 5.999      | 8.065     | 0.9235    | 0.8811 | 0.9387 | 0.5776   | 43.70%              | -2.40%           |
| 3     | yolov8-RepVGG                            | reparameter   | RepVGG                        | 11.312     | 14.547    | 0.9389    | 0.9223 | 0.9616 | 0.6132   | -1.55%              | -0.02%           |
| 4     | yolov8-super-token-sampling              | backbone      | SToken  Attention             | 10.347     | 13.598    | 0.9378    | 0.9193 | 0.9605 | 0.61     | 5.08%               | -0.14%           |
| 5     | yolov8-iRMB                              | backbone      | iRMB                          | 10.354     | 14.043    | 0.94      | 0.9221 | 0.9621 | 0.6148   | 1.97%               | 0.03%            |
| 6     | yolov8-AIFI                              | backbone      | AIFI                          | 10.531     | 13.963    | 0.9398    | 0.9199 | 0.9604 | 0.614    | 2.53%               | -0.15%           |
| 7     | yolov8-efficient-multiscale-attention    | head          | EMA                           | 11.19      | 14.687    | 0.9393    | 0.9206 | 0.9615 | 0.6149   | -2.53%              | -0.03%           |
| 8     | yolov8-SpatialChannelReconstructionConvolution | backbone      | ScConv                        | 11.166     | 14.516    | 0.9406    | 0.9215 | 0.9617 | 0.6152   | -1.33%              | -0.01%           |
| 9     | yolov8-LSK                               | backbone      | LSK                           | 11.171     | 14.544    | 0.9395    | 0.9227 | 0.9616 | 0.6171   | -1.53%              | -0.02%           |
| 10    | yolov8-SlimNeck                          | head          | SlimNeck                      | 11.171     | 14.544    | 0.9426    | 0.9206 | 0.9623 | 0.6152   | -1.53%              | 0.05%            |
| 11    | yolov8-OMNI                              | backbone+head | OMNI                          | 8.882      | 11.959    | 0.9392    | 0.9171 | 0.9594 | 0.6107   | 16.52%              | -0.25%           |
| 12    | yolov8-CA                                | head          | CA-Attention                  | 11.14      | 14.327    | 0.9421    | 0.9217 | 0.9623 | 0.616    | -0.01%              | 0.05%            |
| 13    | yolov8-SE                                | backbone      | SE-Attention                  | 11.169     | 14.325    | 0.9414    | 0.9242 | 0.9624 | 0.6161   | 0.00%               | 0.06%            |
| 14    | yolov8-CBAM                              | head          | CBAM-Attention                | 11.153     |           |           |        |        |          |                     |                  |
| 15    | yolov8-ECA                               | head          | ECA-Attention                 | 11.136     | 14.325    | 0.9402    | 0.9231 | 0.962  | 0.617    | 0.00%               | 0.02%            |
| 16    | yolov8-SimAM                             | head          | SimAM-Attention               | 11.136     | 14.325    | 0.94      | 0.9219 | 0.9625 | 0.6164   | 0.00%               | 0.07%            |
| 17    | yolov8-SKAttention                       | backbone      | SK-Attention                  | 33.247     | 23.136    | 0.9383    | 0.9242 | 0.9617 | 0.6169   | -61.51%             | -0.01%           |
| 18    | yolov8-DoubelAttention                   | backbone      | Doubel-Attention              | 18.647     | 35.77     | 0.9391    | 0.9182 | 0.9576 | 0.6064   | -149.70%            | -0.44%           |
| 19    | yolov8-CoTAttention                      | backbone      | CoT-Attention                 |            |           |           |        |        |          |                     |                  |
| 20    | yolov8-EffectiveSE                       | backbone      | EffectiveSE-Attention         | 18.649     | 35.672    | 0.8732    | 0.57   | 0.7047 | 0.505    | -149.02%            | -26.73%          |
| 21    | yolov8-GlobalContext                     | backbone      |                               |            |           |           |        |        |          |                     |                  |
| 22    | yolov8-GatherExcite                      | backbone      |                               |            |           |           |        |        |          |                     |                  |
| 23    | yolov8-MHSA                              | backbone      | MHSA-Attention                | 19.172     | 35.982    | 0.93963   | 0.9198 | 0.9588 | 0.60808  | -151.18%            | -0.31%           |
| 25    | yolov8-SpatialGroupEnhance               | backbone      | SpatialGroupEnhance-Attention | 18.384     | 35.665    | 0.9416    | 0.9214 | 0.96   | 0.6105   | -148.97%            | -0.19%           |
| 34    | yolov8-SPD                               | backbone      | SPD                           | 12.181     | 21.282    | 0.9413    | 0.9304 | 0.9657 | 0.6239   | -48.57%             | 0.41%            |
| 35    | yolov8-BiFPN                             | head          | BiFPN                         | 11.204     | 14.43     | 0.9432    | 0.9183 | 0.9616 | 0.6164   | -0.73%              | -0.02%           |
| 36    | yolov8-4head                             | head          |                               |            |           |           |        |        |          |                     |                  |
| 37    | yolov8-SPD-4head                         | backbone+head | SPD-4head                     | 11.768     | 25.748    | 0.9466    | 0.9376 | 0.9707 | 0.6287   | -79.74%             | 0.93%            |
| 38    | yolov8-BiFPN-4head                       | head          | BiFPN-4head                   |            |           |           |        |        |          |                     |                  |
| 39    | sahi(切片辅助超推理)                            |               |                               |            |           |           |        |        |          |                     |                  |
| 40    | yolov8-MPDIoU                            | loss          | MPDIoU                        | 11.136     | 14.325    | 0.9401    | 0.9223 | 0.9621 | 0.6178   | 0.00%               | 0.03%            |
| 41    | yolov8-SIoU                              | loss          | SIoU                          | 11.136     | 14.325    |           |        |        |          |                     |                  |
| 42    | yolov8-EIoU                              | loss          | EIoU                          | 11.136     | 14.325    |           |        |        |          |                     |                  |
| 43    | yolov8-WIoU                              | loss          | WIoU                          | 11.136     | 14.325    |           |        |        |          |                     |                  |
| 44    | yolov8-FocalCIoU                         | loss          | FocalCIoU                     | 11.136     | 14.325    | 0.9433    | 0.9208 | 0.9608 | 0.6166   | 0.00%               | -0.10%           |
| 45    | yolov8-FocalSIoU                         | loss          | FocalSIoU                     | 11.136     | 14.325    |           |        |        |          |                     |                  |
| 46    | yolov8-AFPN                              | head          | AFPN                          | 11.649     | 16.749    | 0.9425    | 0.9197 | 0.962  | 0.6176   | -16.92%             | 0.02%            |
| 48    | yolov8-CARAFE                            | head          | CARAFE                        | 11.609     | 15.263    | 0.9392    | 0.9235 | 0.9619 | 0.6169   | -6.55%              | 0.01%            |
| 49    | yolov8-Lion                              | optimizer     | Lion                          | 11.136     | 14.325    | 0         | 0      | 0      | 0        | 0.00%               | -100.00%         |



以下试验标记为ignore为放弃的（算子太奇葩或者改完后模型太大），标记为todo的有时间会继续完成：

| mark   | model                                   | variable type | variable name     | Params(Mb) | FLOPs(Gb) | precision | recall | mAP50 | mAP50-95 | FLOPs reudce  ratio | mAP50 gain ratio |
| ------ | --------------------------------------- | ------------- | ----------------- | ---------- | --------- | --------- | ------ | ----- | -------- | ------------------- | ---------------- |
| ignore | yolov8-DynamicSnake                     | backbone+head | DynamicSnakeConv  |            |           |           |        |       |          |                     |                  |
| ignore | yolov8s-BiFormer                        |               |                   |            |           |           |        |       |          |                     |                  |
| ignore | yolov8s-BGF                             | head          | BGF               | too  big   |           |           |        |       |          |                     |                  |
| todo   | yolov8-NAMAttention                     | backbone      |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-ParNetAttention                  |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-S2Attention                      |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-CrissCrossAttention              |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-CoordAtt                         |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-GAMAttention                     |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-ParallelPolarizedSelfAttention   |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-SequentialPolarizedSelfAttention |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-Triplet                          | backbone      | Triplet-Attention |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-EVC                              |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-GhostNet                         |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-MobileNetV3                      |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-SwinTransformer                  |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-ConvNext                         |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-EfficientNet                     |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-FasterNet                        |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-GhostNetV2                       |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-MobileViT                        |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-ShuffleNetv2                     |               |                   |            |           |           |        |       |          |                     |                  |
| todo   | yolov8-VanillaNet                       |               |                   |            |           |           |        |       |          |                     |                  |