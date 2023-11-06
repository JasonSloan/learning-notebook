# trtpy inference example
记录一下基于trtpy库使用C++版TensorRT推理超分辨率模型Real_ESRGAN的历程。

历程由前到后就是按照01-15序号进行的。

模型权重文件全部未上传

复现每个序号的历程只需要修改Makefile文件中的cuda_home、nano_home、syslib、cpp_pkg改为当前环境下的cuda(trtpy)、NanoLog、sys(trtpy)、cpp(trtpy)家目录，然后将每个序号文件夹下的inference.cpp、inference.hpp、main.cpp替换掉**完整代码**文件夹下对应的文件，make run即可执行。
