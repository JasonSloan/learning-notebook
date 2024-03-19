# cmake最低版本要求
cmake_minimum_required(VERSION 3.0)
# 如果是想debug代码的话，必须加上这一行
set(CMAKE_BUILD_TYPE Debug)
# 工程名称
project(mainproject)
# add_subdirectory(dlib dlib_build)
# 寻找OpenCV库
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
# 包含头文件
include_directories(${OpenCV_INCLUDE_DIRS} include utils src)
# 添加可执行文件
aux_source_directory(src SRC)
aux_source_directory(utils UTILS)
add_executable(${PROJECT_NAME} ${SRC} ${UTILS})
# 设置编译选项
set(CMAKE_CXX_STANDARD 11)
# 链接 OpenCV 库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} openvino::runtime)
# 设置可执行文件的输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

# 打印调试信息
# message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
# message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")