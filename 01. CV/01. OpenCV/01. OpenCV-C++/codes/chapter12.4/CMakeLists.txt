# cmake最低版本要求
cmake_minimum_required(VERSION 2.8)
# 工程名称
project(pro)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
# 寻找OpenCV库
find_package(OpenCV REQUIRED)
# 包含头文件
include_directories(${OpenCV_INCLUDE_DIRS} include/)
# 添加可执行文件
add_executable(${PROJECT_NAME} src/main.cpp)
# 设置编译选项
set(CMAKE_CXX_STANDARD 11)
# 链接 OpenCV 库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
# 打印调试信息
# message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
# message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")