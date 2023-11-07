### 一. 安装segment-anything

```bash
cd segment-anything
pip install -e .
```

### 二. 下载模型权重文件

链接：https://pan.baidu.com/s/17-rXAjEtuz7muf4-nRI8dQ 
提取码：xiio 
--来自百度网盘超级会员V1的分享

放在models文件夹下

### 三. 安装一些库

numpy、opencv-python、torch、shutil、glob、matplotlib==3.5.3

### 四. 注意

实现半自动标注的代码都在auto_label_with_sam.py中，其他的文件夹和文件不用管