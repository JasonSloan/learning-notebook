# 0、opencv解决路径表

| 解决目标            | 采用方法                               | API接口                              |
| --------------- | ---------------------------------- | ---------------------------------- |
| 显示边缘            | 自适应阈值二值化                           | adaptiveThreshold                  |
| Canny           | Canny                              |                                    |
| 计算连通域（一般用于缺陷检测） | Canny+connectedComponentsWithStats | Canny，connectedComponentsWithStats |
| 显示白色边缘          | 形态学梯度                              | morphologyEx                       |
| 显示（绘制）轮廓        | findContours、drawContours          |                                    |
| 基于轮廓计算面积、周长     | contourArea、arcLength              |                                    |
| 获取最大矩形边框        | boundingRect                       |                                    |
| 获取最小外接矩形        | minAreaRect                        |                                    |
| 去噪但模糊           | 高斯模糊+大津法二值化                        | GaussianBlur+threshold             |
| 高斯模糊            | GaussianBlur                       |                                    |
| 均值滤波            | blur                               |                                    |
| 中值滤波            | medianBlur                         |                                    |
| 去噪不模糊           | 双边滤波                               | bilateralFilter                    |
| 取前景、取背景         | 二值化+位运算                            | threshold+bitwise_and              |
| 去除白色噪点          | 腐蚀                                 | erode                              |
| 开运算             | morphologyEx                       |                                    |
| 去除黑色噪点          | 膨胀                                 | dilate                             |
| 闭运算             | morphologyEx                       |                                    |
| 获取白色噪点          | 顶帽                                 | morphologyEx                       |
| 获取黑色噪点          | 黑帽                                 | morphologyEx                       |
| 平衡图像亮度          | 直方图均衡化、自适应直方图均衡                    | equalizeHist、createCLAHE           |

# 一、 图片与视频读取（摄像头调用）（注意opencv中所有读取的图片数据都是uint8格式的）

```python
# 读取图片，imread:默认返回结果是: [H,W,C]；当加载的图像是三原色图像的时候，默认返回的通道顺序是: BGR
# 当设置为0表示灰度图像加载，1表示加载BGR图像, 默认为1，-1表示加载alpha透明通道的图像。
img = cv2.imread(img_path,1)
# 创建窗口，如果设置cv2.WINDOW_AUTOSIZE，则不允许修改窗口大小
window = cv2.namedWindow("window", cv2.WINDOW_NORMAL)
# 更改窗口大小
cv2.resizeWindow("window", 200, 400)
# 让图像暂停delay毫秒，当delay秒设置为0的时候，表示永远; 当键盘任意输入的时候，结束暂停
# waitKey返回键盘按键的ascii值，ord是python中计算ascii的函数
key = cv2.waitKey(0)
# 展示窗口
cv2.imshow("window", 0)
if key == ord("q"):
    # 如果按键按了q，那么就销毁窗口
    cv2.destroyAllWindows()
# 保存图片
cv2.imwrite("./123.png", img)
################################################################################
################################################################################
# 设置像素值
# 调用摄像头并加载视频全流程
# 定义一个窗口
cv2.namedWindow("video", cv2.WINDOW_NORMAL)
# 将窗口resize到640*480的大小
cv2.resizeWindow("video", 640, 480)
# 实例化一个摄像头类，0代表第0个摄像头（有的设备可能有多个摄像头）
# 如果调用摄像头失败，则程序会自动退出
cap = cv2.VideoCapture(0)
# 设置摄像头相关参数（但是实际参数会进行稍微的偏移）
success=cap.set(cv2.CAP_PROP_FRAME_WIDTH, 880)
if success:
    print("设置宽度成功")
success=cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if success:
    print("设置高度成功")
while cap.isOpened():
    # 类似于文件的打开，打开后每次read就是读取一帧，返回值ret为布尔值，代表是否读取成功，frame代表读取到的一帧的信息
    ret, frame = cap.read()
    # 判断如果没有读取到内容，则break进入下一次循环
    if not ret:
        break
    # 将每次读取到的结果使用imshow展示出来
    cv2.imshow("video", frame)
    # 每一帧展示时间为10ms
    key = cv2.waitKey(10)
    # 设置退出方法为使用按键"q"
    if key == ord("q"):
        break
# 释放资源
cap.release()
# 摧毁窗口
cv2.destroyAllWindows()
################################################################################
################################################################################

# 保存摄像机的视频流
# 创建一个基于摄像头的视频读取流，给定基于第一个视频设备
capture = cv.VideoCapture(0)

# 设置摄像头相关参数（但是实际参数会进行稍微的偏移）
success=capture.set(cv.CAP_PROP_FRAME_WIDTH, 880)
if success:
    print("设置宽度成功")
success=capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
if success:
    print("设置高度成功")

# 打印属性
size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
print(size)
# 此时摄像头的帧率(摄像头图像数据没有产生，没办法指导帧率)
print(capture.get(cv.CAP_PROP_FPS))

# 创建一个视频输出对象
# 设置视频中的帧率，也就是每秒存在多少张图片
fps = 15
video_writer = cv.VideoWriter('v1.avi', cv.VideoWriter_fourcc('I', '4', '2', '0'),
                               fps, size)

# 构建10秒的图像输出
num_frames_remaining = 10 * fps - 1
success, frame = capture.read()
while success and num_frames_remaining > 0:
    video_writer.write(frame)
    success, frame = capture.read()
    num_frames_remaining -= 1

# 释放资源
capture.release()
cv.destroyAllWindows()
################################################################################
################################################################################

# 打开视频 ；同打开摄像头的方法，只不过在cv2.VideoCapture(0)把0替换为视频路径
################################################################################
################################################################################

# 调用摄像头有保存摄像头录制到的视频
# 实例化一个类
cap = cv2.VideoCapture(0)
# 指定视频资源保存的格式，此保存格式要和命名文件"output.mp4"的后缀保持一致
# 其他格式"XVID"为avi格式
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# 实例化一个writer，第一个参数为文件名，第二个参数为上面定义的fourcc，第三个参数为每秒保存多少帧，第四个参数为保存的像素尺寸大小
vw = cv2.VideoWriter("output.mp4", fourcc, 30, (640, 480))
while cap.isOpened():
    # 每次读取一帧
    ret, frame = cap.read()
    # 如果读取到的为False，就break
    if not ret:
        break
    # 写入读取到的图片
    vw.write(frame)
    # 将读取到的图片展示出来（也可以不展示）
    cv2.imshow("frame", frame)
    # 按q退出循环，这里waitkey里面设置1，所以实际是每隔1ms就保存一帧图片
    if cv2.waitKey(1) == ord("q"):
        break
# 释放摄像头资源
cap.release()
# 释放VideoWriter资源
vw.release()
```

# 二、图像中像素值的取值、通道的分割合并

```python
# 基于Image对象获取对应的像素值
print("位置(250,300)对应的像素的蓝色取值为:{}".format(img.item(250,300,0)))
################################################################################
################################################################################
# 图像通道的分割和合并
b,g,r = cv.split(img)
img = cv.merge((r, g, b))  # 将原来的r当成新图像的中b，将原来的b当成新图像中的r
```

# 三、图片外围填充（copyMakeBorder）

```python
# 开始添加边框
# 直接复制
replicate = cv.copyMakeBorder(img, top=10, bottom=10, left=10, right=10, borderType=cv.BORDER_REPLICATE)
# 边界反射
reflect = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REFLECT)
# 边界反射，边界像素不保留
reflect101 = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_REFLECT_101)
# 边界延伸循环
wrap = cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_WRAP)
# 添加常数
constant= cv.copyMakeBorder(img,10,10,10,10,cv.BORDER_CONSTANT, value=[128,128,128])
```

# 四、鼠标操作

```python
# 控制鼠标
# 函数参数必须是5个:event鼠标事件；x, y鼠标坐标；flags：鼠标组合按键的值；userdata：每次事件发生后产生的值
def mouse_callback(event, x, y, flags, userdata):
    # event代表的是鼠标事件，每个事件都由不同的数字代号
    if event == 2:
        print("已点击鼠标右键")
# 命名窗口
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
# resize窗口
cv2.resizeWindow("window", 640, 360) # 注意这里的参数顺序为宽、高
# 设置回调函数
cv2.setMouseCallback("window", mouse_callback, "124")
# 设置一个全黑的图片，注意np中传参顺序为行、列，所以要和宽高反过来
img = np.zeros((360, 640, 3), dtype=np.uint8)
while True:
    # 展示图片
    cv2.imshow("window", img)
    # 每次循环已加载的图片都等待10毫秒，如果按键为q，就break
    if cv2.waitKey(10) == ord("q"):
        break
cv2.destroyAllWindows()
# 鼠标事件: 
# - EVENT_MOUSEMOVE   0     鼠标移动
# - EVENT_LBUTTONDOWN   1   按下鼠标左键
# - EVENT_RBUTTONDOWN   2  按下鼠标右键
# - EVENT_MBUTTONDOWN  3 按下鼠标中键
# - EVENT_LBUTTONUP    4      左键释放
# - EVENT_RBUTTONUP   5      右键释放
# - EVENT_MBUTTONUP   6     中键释放
# - EVENT_LBUTTONDBLCLK 7 左键双击
# - EVENT_RBUTTONDBLCLK  8 右键双击
# - EVENT_MBUTTONDBLCLK  9 中键双击
# - EVENT_MOUSEWHEEL  10 鼠标滚轮上下滚动
# - EVENT_MOUSEHWHEEL 11 鼠标左右滚动
#
# flags:
# - EVENT_FLAG_LBUTTON    1  按下左键
# - EVENT_FLAG_RBUTTON    2  按下右键
# - EVENT_FLAG_MBUTTON   4 按下中键
# - EVENT_FLAG_CRTLKEY    8   按下ctrl键
# - EVENT_FLAG_SHIFTKEY   16  按下shift键
# - EVENT_FLAG_ALTKEY       32  按下alt键
```

# 五、使用trackbar

```python
# 使用track_bar
# 设置回调函数，函数参数必须是1个，即BGR中三通道调整后的数值
def trackbar_callback(value):
    print(value)
# 命名窗口
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
# resize窗口
cv2.resizeWindow("window", 640, 360) # 注意这里的参数顺序为宽、高
# 创建trackbar
cv2.createTrackbar("B", "window", 0, 255, trackbar_callback)
cv2.createTrackbar("G", "window", 0, 255, trackbar_callback)
cv2.createTrackbar("R", "window", 0, 255, trackbar_callback)
# 设置一个全黑的图片，注意np中传参顺序为行、列，所以要和宽高反过来
img = np.zeros((360, 640, 3), dtype=np.uint8)
while True:
    # 得到拖动后trackbar对应通道的返回值
    b = cv2.getTrackbarPos("B", "window")
    g = cv2.getTrackbarPos("G", "window")
    r = cv2.getTrackbarPos("R", "window")
    # 将返回值赋值给img
    img[:] = [b, g, r]
    # 展示图片
    cv2.imshow("window", img)
    # 设置退出按键为q
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
```

#  六、颜色空间转换

```python
# 颜色空间转换列表
color_space = [cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2HLS, cv2.COLOR_BGR2YUV, cv2.COLOR_BGR2YCrCb]
# 颜色空间转换
for i in color_space:
    pic = cv2.cvtColor(img, i)
    cv2.imshow("pic", pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 图像通道分割与融合
# 分割
b, g, r = cv2.split(img)
# 融合
img_merge = cv2.merge([b, g, r])
```

# 七、画线操作

```python
# 画线
cv2.line(img, pt1=(0,0), pt2=(511,511), color=(255,0,0), thickness=5)
# 画矩形
cv2.rectangle(img, pt1=(10,10), pt2=(50,320), color=(255,0,0), thickness=5)
# 画圆
cv2.circle(img, center=(200,200), radius=100, color=(0,0,255), thickness=1)
# 画椭圆
cv2.ellipse(img, center=(210,310), axes=(100,50), angle=0, startAngle=0, endAngle=180,
           color=(255,0,0), thickness=5)
# 绘制多边形
# 注意pts点集必须是三维的
pts = np.array([[[0,0], [100, 200], [200, 300], [300, 10]]], np.int32)
cv2.polylines(img, pts, True, (0, 0, 255), 2, 16)
# 绘制文本，不支持中文
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, text='OpenCV', org=(10,450), fontFace=font,
           fontScale=4, color=(255,255,255), thickness=2, lineType=cv.LINE_AA)
cv2.imshow("line", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#  八、图片融合

```python
# 图片运算，加减乘除
img1 = cv2.resize(img1, (480, 640))
img2 = cv2.resize(img2, (480, 640))
img = cv2.add(img1, img2)
img_weighted = cv2.addWeighted(img1, 0.2, img2, 0.8)
```

#  九、位运算（取前景、取背景）

```python
# 结合位运算和二值化，可以取一张图片的前景和另一张图片的背景，从而完美融合两张图片

# 图像的位运算（将logo放到图像的右上角）
# 加载图像
img1 = cv.imread("xiaoren.png")
img2 = cv.imread("opencv-logo.png")

# 获取一个新数据（右上角区域数据）
rows1, cols1, _ = img1.shape
rows2, cols2, channels = img2.shape
start_rows = 50
end_rows = rows2 + start_rows
start_cols = cols1 - cols2 - 200
end_cols = cols1 - 200
roi = img1[start_rows:end_rows, start_cols:end_cols]

# 将图像转换为灰度图像
img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
# 将灰度图像转换为黑白图像，做一个二值化操作，得到蒙版mask（用来取前景，只有前景的位置为255，其余位置都为0）
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
# 对图像做一个求反的操作，即255-mask，得到蒙版mask_inv（用来取背景，只有背景的地方为255，其余地方都为0）
mask_inv = cv.bitwise_not(mask)
# 获取得到前景图
img2_fg = cv.bitwise_and(img2,img2, mask=mask)
# 获取得到背景图（对应mask_inv为True的时候，进行and操作，其它位置直接设置为0）
# 在求解bitwise_and操作的时候，如果给定mask的时候，只对mask中对应为1的位置进行and操作，其它位置直接设置为0
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
# 前景颜色和背景颜色合并
dst = cv.add(img1_bg,img2_fg)
# dst = img1_bg + img2_fg

# 复制粘贴
img1[start_rows:end_rows, start_cols:end_cols] = dst

# 可视化
cv.imshow('res',img1)
cv.waitKey(0)
cv.destroyAllWindows()
```

#  十、仿射变换（缩放翻转旋转平移透视）(本质是对图像做矩阵乘法)

```python
# 图像缩放
# 按照新尺寸进行缩放
img1 = cv2.resize(img1, (400, 600))
# 按照与原尺寸的比率进行缩放
img1 = cv2.resize(img1, dsize=None, fx=0.5, fy=0.8, interpolation=cv2.INTER_AREA)

# 图像翻转，1左右翻转，0上下翻转
img1 = cv2.flip(img1, 1)

# 图像旋转，顺时针旋转90度
img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)

# 图像旋转
h, w, c = img1.shape
# 定义仿射变换矩阵，以(100, 100)为中心点旋转15度，缩放比率为1
M = cv2.getRotationMatrix2D(center=(100, 100), angle=45, scale=1)
img1 = cv2.warpAffine(src=img1, M=M, dsize=(w, h)，borderValue=[0,0,0])



# 图像平移
# 仿射变换不改变图片尺寸
h, w, c = img1.shape
# 定义变换矩阵M，此例子为将原图向右横向移动100，向下纵向移动50
M = np.array([[1, 0, 100], [0, 1, 50]], dtype=np.float32)
img1 = cv2.warpAffine(img1, M, (w, h))

# 图像透视变换
# 从原图中截取出来的四个点坐标（坐标顺序左上角，右上角，左下角，右下角）
# 此四个点坐标可能超过原图尺寸越界，越界的部分默认用0填充
src = np.float32([[0, 0], [400, 0], [0, 440], [400, 440]])
# dst意味着将原图截取后再缩放至dst大小尺寸
dst = np.float32([[0, 0], [200, 0], [0, 300], [200, 300]])
M = cv2.getPerspectiveTransform(src=src, dst=dst)
img1 = cv2.warpPerspective(src=img1, M=M, dsize=[200, 300])
```

# 十一、滤波（卷积）（去噪磨皮）(本质是对图像做卷积)

```python
# 普通滤波
# 指定卷积核，kernel1可以使图片变虚；kernel2可以提取图片的边缘信息
kernel1 = np.ones((3,3), dtype=np.float32) / 9  # 保持期望不变
kernel2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
# ddepth=-1代表输出通道数和输入通道数一样。默认卷积步长为1，使用padding，保持图片尺寸不变
img1 = cv2.filter2D(src=img1, ddepth=-1, kernel=kernel2)

"""高斯滤波：适用于处理图片中的噪声点（去噪）
高斯滤波，使用3*3卷积核，卷积核中的9个数符合高斯分布，卷积核中心点数值最大（概率最大），距离中心点越远数值越小（概率越小）。卷积核中所有元素之和期望为1。sigmaX代表的是横轴方向的方差值，方差值越大，模糊效果越强。"""
img1 = cv2.GaussianBlur(src=img1, ksize=(3,3), sigmaX=10)

# 中值滤波：适用于去除椒盐噪点
img1 = cv2.medianBlur(img1, ksize=3)

# 双边滤波：既能去噪又能保留边缘信息，不会将边缘模糊化
# d类似于卷积核大小，d越大磨皮效果越好（实际是以当前被计算像素点为圆心，去d为直径内的像素计算高斯）
# simgaColor可以理解为保留边缘信息的反权重，值越大，边缘越模糊
# 如果指定了d，那么sigmaSpace可能没啥效果
img1 = cv2.bilateralFilter(img1, d=5, sigmaColor=20,sigmaSpace=100)
```

# 十二、边缘检测（Canny)

```python
Canny
Canny算法是一种比Sobel和Laplacian效果更好的一种边缘检测算法；在Canny算法中，主要包括以下几个阶段：
Noise Reduction：降噪，使用5*5的kernel做Gaussian filter降噪；
Finding Intensity Gradient of the Image：求图像像素的梯度值；
Non-maximum Suppression：删除可能不构成边缘的像素，即在渐变方向上相邻区域的像素梯度值是否是最大值，如果不是，则进行删除。
Hysteresis Thresholding：基于阈值来判断是否属于边；大于maxval的一定属于边，小于minval的一定不属于边，在这个中间的可能属于边的边缘。

参考页面：
https://en.wikipedia.org/wiki/Canny_edge_detector
http://dasl.unlv.edu/daslDrexel/alumni/bGreen/www.pages.drexel.edu/_weg22/can_tut.html
# Canny算法边缘检测
# threshold1代表边缘梯度值的下限，小于该下限的不算边缘。所以threshold1越大，对边缘要求越宽松，检测到的边缘越多。
# threshold2代表边缘梯度值的上限，大于该上限的算边缘。所以threshold2越大，对边缘要求越严格，检测到的边缘越少。
img1 = cv2.Canny(img1, threshold1=50, threshold2=100)
```

# 十三、计算连通域

```python
canny_image = cv2.Canny(image, 80, 130)
# cv2.imshow("canny_image",canny_image)
# cv2.waitKey(0)
"""
num_labels：所有连通域的数目
labels：图像上每一像素的标记，用数字1、2、3…表示（不同的数字表示不同的连通域）
stats：每一个标记的统计信息，是一个5列的矩阵，每一行对应每个连通区域的外接矩形的x、y、width、height和面积
centroids：连通域的中心点
"""
retval, labels, stats, centroids = cv2.connectedComponentsWithStats(canny_image, connectivity=8)
```

# 十四、腐蚀、膨胀、开闭操作、顶帽黑帽

```python
#　腐蚀 : 黑色腐蚀白色
# 使用3*3的卷积腐蚀2次
img2 = cv2.erode(img1, kernel=(3, 3), iterations=2)

# 膨胀 ：白色膨胀
# 使用3*3的卷积膨胀2次。一般先腐蚀再膨胀，可以去掉白色的细须。
img2 = cv2.dilate(img2, kernel=(3, 3), iterations=2)

# 开运算=先腐蚀再膨胀
dst = cv.morphologyEx(img, op=cv.MORPH_OPEN, kernel=kernel, iterations=1)

# 闭运算=先膨胀再腐蚀
dst = cv.morphologyEx(img, op=cv.MORPH_CLOSE, kernel=kernel, iterations=1)

# 形态学梯度
# MORPH_GRADIENT代表形态学梯度，等于原图减去腐蚀后的图，可以返回原图中所有白色与黑色交接处的轮廓。
img2 = cv2.morphologyEx(img2, op=cv2.MORPH_GRADIENT, kernel=(3, 3), iterations=10)

# 顶帽：等于原图减去开运算，也就是得到了图像的白色噪点，cv2.MORPH_TOPHAT
# 黑帽：等于原图减去闭运算，也就是得到了图像的黑色噪点，cv2.MORPH_BLACKHAT
dst = cv2.morphologyEx(img2, op=cv2.MORPH_BLACKHAT, kernel=(3, 3), iterations=2)
```

# 十五、阈值二值化（蒙版，注意蒙版维度必须是H*W）

```python
# 全局阈值二值化（将整张图片所有的数值要么变成255，要么变成0）
# 要先将图片转化为灰度图
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# 进行普通二值化操作(第一个参数是返回的阈值，第二个参数返回的是二值化之后的图像)
# 普通二值化操作， 将小于等于阈值thresh的设置为0，大于该值的设置为maxval
ret, thresh1 = cv.threshold(src=img, thresh=127, maxval=255, type=cv.THRESH_BINARY)
# 反转的二值化操作， 将小于等于阈值thresh的设置为maxval，大于该值的设置为0
ret, thresh2 = cv.threshold(src=img, thresh=127, maxval=255, type=cv.THRESH_BINARY_INV)
# 截断二值化操作，将小于等于阈值thresh的设置为原始值，大于该值的设置为maxval
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
# 0二值化操作，将小于等于阈值的设置为0，大于该值的设置为原始值
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
# 反转0二值化操作，将小于等于阈值的设置为原始值，大于阈值的设置为0
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
# 大津法二值化操作
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
################################################################################
################################################################################
# 转换颜色空间
# 加载数据
img = cv.imread('opencv-logo.png')

# 转换为HSV格式
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# 定义像素点范围
# 在hsv中蓝色的范围
# lower = np.array([100,50,50])
# upper = np.array([130,255,255])
# 红色的范围
lower = np.array([150,50,50])
upper = np.array([200,255,255])

# 在这个范围的图像像素设置为255，不在这个范围的设置为0（蒙版）
mask = cv.inRange(hsv, lower, upper)
# 进行And操作进行数据合并
dst = cv.bitwise_and(img,img, mask= mask)

# 图像可视化
cv.imshow('hsv', hsv)
cv.imshow('mask', mask)
cv.imshow('image', img)
cv.imshow("dest", dst)
cv.waitKey(0)
cv.destroyAllWindows()
################################################################################
################################################################################
# 进行自适应二值化操作
# 因为二值化操作的时候需要给定一个阈值，但是实际情况下阈值不是特别好给定的。
# 所以可以基于本身的图像数据，根据当前区域的像素值获取适合的阈值对当前区域进行二值化操作
# 使用均值的方式产生当前像素点对应的阈值，
# 使用(x,y)像素点邻近的blockSize*blockSize区域的均值寄减去C的值
gray = cv2.adaptiveThreshold(gray, maxValue=200, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             thresholdType=cv2.THRESH_BINARY, blockSize=3, C=0)
# 使用高斯分布的方式产生当前像素点对应的阈值
# 使用(x,y)像素点邻近的blockSize*blockSize区域的加权均值寄减去C的值，
# 其中权重为和当前数据有关的高斯随机数
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv.THRESH_BINARY, 11, 2)
```

# 十五、轮廓相关

```python
# 绘制轮廓
# 画轮廓必须使用灰度图然后转化为二值
gray = cv2.cvtColor(img2, code=cv2.COLOR_BGR2GRAY)
thresh, binary = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
# 返回值contours是一个列表，列表中每个元素是一个矩阵（也就是一个轮廓），矩阵中的每个元素就是轮廓中的点
# mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE是最常用用法不用更改
contours, result = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
print(contours[0].shape)  # (4, 1, 2)第一个轮廓有4个点，并且每个点由 (x, y) 坐标表示。注意在这个例子中，维度3是省略的，因为是灰度图像，每个点只需要 (x, y) 坐标来表示
# 因为drawContours方法会改变原图，所以要将原图copy一份再画轮廓
new_img = img2.copy()
# 绘制轮廓：contourIdx=-1即代表绘制所有轮廓
cv2.drawContours(new_img, contours=contours, contourIdx=-1, color=(0, 0, 255))

# 计算轮廓面积
area = cv2.contourArea(contours[1])

# 计算轮廓周长
perimeter = cv2.arcLength(contours[1], closed=False)

# 多边形逼近轮廓
# 画轮廓必须使用灰度图然后转化为二值
gray = cv2.cvtColor(img1, code=cv2.COLOR_BGR2GRAY)
thresh, binary = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
# 返回值contours是一个列表，列表中每个元素是一个矩阵（也就是一个轮廓），矩阵中的每个元素就是轮廓中的点
# mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE是最常用用法不用更改
contours, result = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
# 因为drawContours方法会改变原图，所以要将原图copy一份再画轮廓
new_img = img1.copy()
# 多边形逼近轮廓，即使用多边形逼近曲线。epsilon越小，越逼近曲线。closed代表是否是闭合曲线。
approx = cv2.approxPolyDP(contours[0], epsilon=10, closed=False)
# 绘制轮廓：contourIdx=-1即代表绘制所有轮廓
cv2.drawContours(new_img, contours=[approx], contourIdx=-1, color=(0, 0, 255), thickness=2)

# 绘制凸包
# 画轮廓必须使用灰度图然后转化为二值
gray = cv2.cvtColor(img1, code=cv2.COLOR_BGR2GRAY)
thresh, binary = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
# 返回值contours是一个列表，列表中每个元素是一个矩阵（也就是一个轮廓），矩阵中的每个元素就是轮廓中的点
# mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE是最常用用法不用更改
contours, result = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
# 因为drawContours方法会改变原图，所以要将原图copy一份再画轮廓
new_img = img1.copy()
# 计算凸包
hull = cv2.convexHull(contours[0])
# 绘制凸包
cv2.drawContours(new_img, contours=[hull], contourIdx=-1, color=(0, 0, 255), thickness=2)


# 最小外接矩形
# 画轮廓必须使用灰度图然后转化为二值
gray = cv2.cvtColor(img1, code=cv2.COLOR_BGR2GRAY)
thresh, binary = cv2.threshold(gray, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
# 返回值contours是一个列表，列表中每个元素是一个矩阵（也就是一个轮廓），矩阵中的每个元素就是轮廓中的点
# mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE是最常用用法不用更改
contours, result = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
# 因为drawContours方法会改变原图，所以要将原图copy一份再画轮廓
new_img = img1.copy()
# 计算最小外接矩形
rect = cv2.minAreaRect(contours[0])
# rect的值不符合规则，要因为是坐标要转化为整数值
box = np.round(cv2.boxPoints(rect)).astype(int)
# 绘制最小外接矩形
cv2.drawContours(new_img, contours=[box], contourIdx=-1, color=(0, 0, 255), thickness=2)
```

# 十六、图像金字塔（上采样和下采样）

```python
# 高斯金字塔
# 对图像进行卷积操作，然后暴力去除偶数行偶数列（2倍下采样）
gussian = cv2.pyrDown(img2)
# 上采样：对一张图片将0填充到偶数行、偶数列中，然后将图片中原有的像素值的分配到其余的3个0中，然后再进行乘以4保持期望不变
gussian = cv2.pyrUp(gussian)

# 拉普拉斯金字塔
# 将图片先下采样再上采样，然后用原图减去采样后图片就得到了下采样丢失的信息（可用于图像压缩）
dst = img2 - gussian
```

# 十七、绘制直方图

```python
# 图像直方图
# channels代表的是统计第0维度（BGR中的B）的直方图，mask可以用来截取图像中的一部分，histsize代表横轴从0-255，ranges代表统计像素值范围为0-255
hist = cv2.calcHist([img1], channels=[0], mask=None, histSize=[256], ranges=[0 ,255])
# 绘制图像直方图
import matplotlib.pyplot as plt
plt.hist(hist)
plt.show()
# 将图片转化为灰度图像
img2 = cv2.cvtColor(img2, code=cv2.COLOR_BGR2GRAY)
# 图像均衡化：使图片不会太亮也不会太暗
img2 = cv2.equalizeHist(img2)
```

# 十八、角点检测

```python
# harris(哈里斯)角点检测
# 将图片转化为灰度图
gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
# 进行哈里斯角点检测计算
# blockSize代表每次比较梯度值变化的块大小,ksize代表使用的卷积核大小,k代表一个阈值
# blockSize越小能检测到的角点越多,ksize一般设为3,k一般设为0.04
# 返回的dst是和图片维度相同的一个矩阵,矩阵中存着每个像素的角点响应值
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
# 将dst中最大的角点响应值取出来,按照最大角点响应值的0.01倍作为阈值,大于该阈值的角点筛选出来,然后大于该阈值的角点的值设为红色方便显示
img[dst > (0.01 * dst.max())] = [0, 0, 255]
cv2.imshow("img", img)


# 托马斯角点检测
img = cv2.imread(r"C:\Users\root\Desktop\34.png")
# 将图片转化为灰度图
gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
# 计算角点
# maxCorners代表最多能检测到的角点数,0代表检测所有角点
# qualityLevel代表检测角点质量,越小代表检测的角点越多
# minDistance代表最小距离
# 返回值corners,corners中存储的是每一个角点的坐标
corners = cv2.goodFeaturesToTrack(gray, maxCorners=0, qualityLevel=0.01, minDistance=10).astype(np.int0)
# 遍历每一个坐标,将每一个坐标绘制出来
for i in corners:
    x, y = i.ravel()
    cv2.circle(img, center=(x, y), radius=3, color=(0,0,255))
cv2.imshow("img", img)
```

# 十九、sift关键点检测

```python
# 关键点检测的优点就是不会受图片的仿射变换以及亮度的影响,不管将图片进行怎样的变换,都能准确检测到关键点
img = cv2.imread(r"C:\Users\root\Desktop\4.jpg")
# 将图片转化为灰度图
gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
# 实例化一个sift
sift = cv2.xfeatures2d.SIFT_create()
# 检测灰度图得到关键点
kp = sift.detect(gray))
# 绘制关键点
cv2.drawKeypoints(image=img, keypoints=kp, outImage=img)
cv2.imshow("img", img)
```

