```bash
.

|-- 01. CV

|   |-- 00. 标注

|   |-- 01. OpenCV

|   |-- 02. 目标检测

|   |-- 03. ReID

|   |-- 04. 目标跟踪

|   |-- 05. 超分辨率

|   |-- 06. 相机模型

|   |-- 07. 大模型

|   `-- 08. 特征点匹配

|-- 02. NLP

|   |-- 01. jieba&hanlp

|   `-- 02. Transformer

|-- 03. 推理与优化加速

|   |-- 00. ONNX

|   |-- 01. TensorRT

|   |-- 02. OpenVINO

|   |-- 03. BMNN

|   |-- 04. RKNN

|   `-- 05. 剪枝与重参

|-- 04. 训练

|   |-- 01. 谷歌研究院训练调参指南中文翻译

|   |-- 02. 多卡分布式训练方法

|   |-- 03. 日志

|   |-- 04. Python计时代码

|   |-- 05. 数据集划分

|   |-- 06. 彩色打印

|   |-- 07. 初始化随机数种子

|   |-- 08. 参数解析代码argparse

|   |-- 09. 各种学习率

|   |-- 10. 自增目录

|   |-- 11. 滑动平均模型ModelEMA

|   |-- 12. EarlyStop

|   |-- 13. 标签平滑LabelSmoothing

|   |-- 14. 各种损失函数

|   |-- 15. 多尺度MuiltiScale

|   |-- 16. 各种评估指标

|   |-- 17. letterbox

|   |-- 18. 各种距离

|   |-- 19. 标注格式转换脚本

|   |-- 20. 快速漂亮的打印模型各个层以及参数信息

|   |-- 21. 正负样本分配策略

|   |-- 22. 将NMS后处理集成到网络中

|   `-- 23. nvitop

|-- 05. 图像相关常用脚本

|   |-- 01. images2video.py

|   |-- 02. video2images.py

|   |-- 03. yuv2png.py

|   |-- 04. image_convert_2_binary.py

|   |-- 05. array_convert_2_binary.py

|   |-- 06. binary_convert_2_image.py

|   |-- 07. base64EncodeDecode.py

|   |-- 08. generate_paired_images_from_videos.py

|   |-- 09. image_rectify.py

|   `-- README.md

|-- 06. Python

|   |-- 00. 常用package

|   |-- 01. Python多进程多线程

|   |-- 02. 使用Python调用command

|   |-- 03. 正则表达式

|   |-- 04. 使用Python下载网络上的文件

|   `-- 05. 装饰器

|-- 07. 框架

|   |-- 00. Numpy

|   `-- 01. PyTorch

|-- 08. C++

|   |-- 00. C++基础

|   |-- 000. C++编译相关

|   |-- 01. 日志

|   |-- 02. 多线程

|   |-- 03. 将C++封装成python可调用的库

|   |-- 04. C++版本的tqdm

|   |-- 04. 文件读取(open)

|   |-- 05. 文件写入(open)

|   |-- 06. 判断一个文件或者文件夹是否存在(os.path.exists)

|   |-- 07. 创建文件夹(mkdir)

|   |-- 08. 获取文件夹下的所有文件名(os.listdir)

|   |-- 09. 获得完整路径中的文件夹路径(os.path.dirname)

|   |-- 10. 获得完整路径中的文件名字(os.path.basename)

|   |-- 11. 去掉字符串两端的空白(.strip)

|   |-- 12. 将指定字符替换为新的字符(.replace)

|   |-- 13. 按照指定符号分割字符串(.split)

|   |-- 14. 判断某字符串是否以什么开头或结尾(.startswith&.endswith)

|   |-- 15. 提取路径中的文件名(os.path.basename)

|   |-- 16. 计时代码

|   |-- 17. 在c中调用cpp代码

|   |-- 18. C++中的回调函数用法

|   |-- 19. C++中的字典

|   |-- 20. C++中两个头文件互相include

|   |-- 21. C++中对函数传入可变长参数

|   |-- 22. 纯头文件使用方式的第三方库json

|   |-- 23. 纯头文件使用方式的第三方库pybind11

|   |-- 24. 纯头文件使用方式的第三方库Eigen

|   |-- 98. 设置Asan编译选项检查内存泄漏

|   `-- 99. 使用gdb调试segmentation fault的代码

|-- 09. linux

|   |-- 00. 一个命令获得当前所有的环境.md

|   |-- 01. windows、linux挂载.md

|   |-- 02. 常用命令.md

|   |-- 03. ubuntu换源.md

|   |-- 04. 修改DNS.md

|   |-- 05. ssh连接总掉线.md

|   |-- 06 arm版本linux安装conda安不上.md

|   |-- 07. root权限依然删除不了某个文件.md

|   |-- 08. ldd查看总有些库链接不上.md

|   |-- 09. 设置某服务开机自启动.md

|   |-- 10. vim下格式化json.md

|   |-- 11. 格式化并挂载硬盘.md

|   `-- assets

|-- 10. Docker

|   `-- README.md

|-- 11. IDE相关

|   |-- 01. vscode

|   |-- 02. Pycharm配置文件settings.zip

|   `-- 03. 谷歌搜索技巧

|-- 12. frp内网穿透

|   |-- README.md

|   `-- assets

|-- 13. git及github

|   |-- 03. git release使用

|   |-- 1. 一个完整的git流程

|   `-- 2.  FastGithub软件

|-- 14. web相关

|   |-- 01. flask

|   `-- 02. curl提交GET、POST请求

|-- 15. 机器学习数学基础

|   `-- 机器学习的数学基础.docx

|-- 16. 面试真题

|   |-- README.md

|   |-- assets

|   `-- 手撕代码

|-- 17. 算法复现

|   |-- 01. Canny

|   |-- 02. ResNet

|   |-- 03. ROIPooling

|   |-- 04. RANSAC

|   |-- 05. CBAM

|   `-- 06. Generalized-Hough

|-- 18. huggingface

|   |-- README.md

|   `-- assets

|-- 19. Latex

|   |-- README.md

|   `-- assets

|-- 20. MarkDown

|   |-- README.md

|   `-- assets

`-- 21. 其他

	`-- 01. 免费看medium文章
```

