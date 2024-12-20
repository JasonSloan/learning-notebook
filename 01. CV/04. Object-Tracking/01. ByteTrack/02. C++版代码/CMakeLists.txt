#####CMakeLists示例######
# cmake最低版本要求
cmake_minimum_required(VERSION 3.0)
# 如果是想debug代码的话，必须加上这一行
set(CMAKE_BUILD_TYPE Debug)
# 工程名称
project(mainproject)
# 寻找OpenCV库
set(OpenCV_HOME opencv-build)
set(EIGEN_HOME eigen-build)
set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_video opencv_videoio)
# 包含头文件
include_directories(include ${OpenCV_HOME}/include/opencv4 ${EIGEN_HOME}/include/eigen3)
# 包含库文件
link_directories(${OpenCV_HOME}/lib)
# 添加可执行文件
file(GLOB_RECURSE SRC "src/*.cpp")
# 设置编译选项
set(CMAKE_CXX_STANDARD 17)
# 设置编译输出, 可执行文件以及动态库 
add_executable(${PROJECT_NAME} ${SRC} ${UTILS})
# 链接一些库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
# 设置可执行文件的输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

