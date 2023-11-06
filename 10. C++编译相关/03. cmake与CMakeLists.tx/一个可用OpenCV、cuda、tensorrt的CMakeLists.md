先安装OpenCV：

sudo apt update

sudo apt-get install -y libopencv-dev

```python
cmake_minimum_required(VERSION 3.12)
project(real-esrgan)

# 加上有用，告诉编译器我的动态库是为了跟别人的动态库一起编译的，用作user API
add_definitions(-DAPI_EXPORTS)
# 不知道干嘛用的
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# Set C++ standard and build type
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CUDA_HOME /home/research/software/anaconda3/envs/esrgan/lib/python3.9/site-packages/trtpy/trt8cuda115cudnn8)
set(OPENCV_HOME /home/research/software/anaconda3/envs/esrgan/lib/python3.9/site-packages/trtpy/cpp-packages/opencv4.2)

# Add compiler flags
add_compile_options(-std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED)

# Include directories for your project
include_directories(include)
include_directories(src)

# Include and link directories for CUDA and TensorRT
include_directories(${CUDA_HOME}/include/tensorRT)
include_directories(${CUDA_HOME}/include/cuda)
include_directories(${OPENCV_HOME}/include)
link_directories(${CUDA_HOME}/lib64)
link_directories(${OPENCV_HOME}/lib)

# Create a shared library instead of an executable
add_library(inference SHARED
    src/inference.cpp           # Your custom-defined inference source file
)

# Add the -fPIC flag to ensure position-independent code
set_property(TARGET inference PROPERTY POSITION_INDEPENDENT_CODE ON)

# Link the necessary libraries to the shared library
target_link_libraries(inference PRIVATE nvinfer cudart opencv_core opencv_imgcodecs opencv_imgproc)
```



