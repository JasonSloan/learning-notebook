# cmake-language
cmake_minimum_required(VERSION 3.12)

project(real-esrgan)

# Set C++ standard and build type
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# Add compiler flags
add_compile_options(-std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED)

# Check if CUDA is available
find_package(CUDA REQUIRED)

if(WIN32)
    enable_language(CUDA)
endif(WIN32)

# Include directories for your project
include_directories(include)
include_directories(src)

# Include and link directories for CUDA and TensorRT
include_directories(/root/miniconda3/lib/python3.8/site-packages/trtpy/trt8cuda115cudnn8/include/tensorRT)
include_directories(/root/miniconda3/lib/python3.8/site-packages/trtpy/trt8cuda115cudnn8/include/cuda)
link_directories(/root/miniconda3/lib/python3.8/site-packages/trtpy/trt8cuda115cudnn8/lib64)

# Create a shared library instead of an executable
add_library(inference SHARED    # 动态库的名字叫inference，也就是会产生一个叫libinference.so的动态库
    src/inference.cpp           # Your custom-defined inference source file
)

# Link the necessary libraries to the shared library
target_link_libraries(inference PRIVATE nvinfer cudart)

# Find OpenCV and include its directories
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Link OpenCV libraries to the shared library
target_link_libraries(inference PRIVATE ${OpenCV_LIBS})

if(UNIX)
    # Add optimization and pthread flags for Unix systems
    add_compile_options(-O2 -pthread)
endif(UNIX)



