## 使用方法

1. 将检测框的文件放入dt下, 格式为cls+xywh+conf; 将真实框的文件放入gt下, 格式为cls+xywh, 这两个文件夹下的边框坐标均为值在0-1之间的相对坐标; 将图片放在images文件夹下。三个文件夹中的文件名要一一对应

2. 将类别名字按序写入class_list.txt文件

3. 使用yolo2voc.py将dt和gt中yolo格式的相对坐标转换为voc格式的绝对坐标(也就是将相对值乘以图像宽高), 输出文件产生在dt-output和gt-output中

4. 克隆代码 

```bash
git clone https://github.com/rafaelpadilla/Object-Detection-Metrics
```

5. 进入Object-Detection-Metrics文件夹中运行pascalvoc.py文件, 并指定参数获取结果

```bash
python pascalvoc.py -gt ../gt-output -det ../dt-output -gtformat xyrb -detformat xyrb
```



