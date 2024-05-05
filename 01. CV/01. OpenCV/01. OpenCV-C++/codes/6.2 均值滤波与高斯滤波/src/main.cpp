#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>


int main(){
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE);
    cv::Mat result;
    // 均值滤波
    cv::blur(image, result, cv::Size(5,5)); // 滤波器尺寸
    cv::imwrite("../workspace/mean.png", result);

    // 高斯滤波
    cv::GaussianBlur(image, result, 
    cv::Size(5,5), // 滤波器尺寸
    1.5); // 控制高斯曲线形状的参数
    cv::imwrite("../workspace/gaussian.png", result);
    printf("Done!\n");
}