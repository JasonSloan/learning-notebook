#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>
#include <iostream>


int main(){
    // OpenCV 提供的计算代码运行时间的方法
    // 计算开始时间
    const int64 start = cv::getTickCount(); 
    // 运行要计算时间的代码...
    // 计算结束时间（要除以CPU频率）
    double duration = (cv::getTickCount() - start) / cv::getTickFrequency();
    }