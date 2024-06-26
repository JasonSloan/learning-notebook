## 1. 推流

到[github官网](https://github.com/bluenviron/mediamtx.git) 的release中下载mediamtx可执行文件拷贝到linux系统中，使用./mediamtx启动rtsp服务。

使用ffmpeg推流，推流命令如下：

```bash
ffmpeg -re -i input.mp4 -c:v libx264 -f rtsp rtsp://192.168.103.241:8554/mystream
# 参数解释
-re: Read input at native frame rate. Mainly used for live streaming.
-i input.mp4: Specify the input file or stream.
-c:v libx264: Encode the video using the H.264 codec.
-f rtsp: Specify the output format as RTSP.
rtsp://localhost:8554/mystream: The RTSP URL where the stream will be pushed.
```

## 2. 使用VLC拉流显示查看

在VLC媒体选项中选择打开网络串流，输入rtsp地址，例如：rtsp://192.168.103.241:8554/mystream，即可显示。

## 3. 使用opencv拉流解码为图片

示例代码：

```python
import cv2
import numpy as np

rtsp_url = 'rtsp://192.168.103.241:8554/mystream'
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break
    
    cv2.imwrite(f'outputs/{count:06d}_.jpg', frame)
    count += 1  
```

