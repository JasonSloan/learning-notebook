
## 编译安装OpenCV

        安装cmake:  apt-get install cmake

        安装依赖: apt-get install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg.dev libtiff5.dev libswscale-dev

        下载OpenCV4.6.0: 链接：https://pan.baidu.com/s/1OuFZNub5VGMsoF8oOSNxjQ  提取码：5tgz

        解压进入opencv4.6.0文件夹,创建build文件夹, 进入build文件夹, 执行cmake, 执行make, 执行make install
        
        如果想将编译后头文件和库文件安装到指定目录, 在cmake编译时需要指定-DCMAKE_INSTALL_PREFIX=/path/to/custom/folder, 然后编译结束后make install就可以安装在指定目录中了
        
        如果出现以下错误:By not providing "FindOpenCV.cmake" in CMAKE_MODULE_PATH this project has
    asked CMake to find a package configuration file provided by "OpenCV", but
    CMake did not find one.
     则需要在CMakeLists.txt中增加一行"set(OpenCV_DIR /path/to/opencv-4.6.0)", 指定opencv源码的根目录
    
        在/etc/ld.so.conf.d下创建文件OpenCV.conf, 写入"/usr/local/lib", 保存退出执行sudo ldconfig, 使opencv的动态库可以被系统链接到
    
        编写main.cpp, 验证opencv是否好用
