 
## 编译安装OpenCV

        安装cmake:  apt-get install cmake

        安装依赖: apt-get install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg.dev libtiff5.dev libswscale-dev

        下载OpenCV4.6.0: 链接：https://pan.baidu.com/s/1OuFZNub5VGMsoF8oOSNxjQ  提取码：5tgz

        解压进入opencv4.6.0文件夹,创建build文件夹, 进入build文件夹, 执行cmake, 执行make, 执行make install

        在/etc/ld.so.conf.d下创建文件OpenCV.conf, 写入"/usr/local/lib", 保存退出执行sudo ldconfig, 使opencv的动态库可以被系统链接到

        编写main.cpp, 验证opencv是否好用
