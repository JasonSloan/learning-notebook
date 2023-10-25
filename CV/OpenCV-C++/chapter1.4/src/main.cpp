#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>

int main(){
    cv::Mat image;
    image = cv::imread("../workspace/dog.png", cv::IMREAD_COLOR);
    cv::Mat gray(image.size(),CV_8U, cv::Scalar(50)); // 50代表图像的初始值
    cv::Mat image_copy;
    cv::Mat image_ = image; // 浅拷贝，image_和image指向同一块内存
    image_copy = image.clone(); // 深拷贝
    image.convertTo(image, CV_32F, 1/255.0, 0.0); // 将图像转换为浮点型，1/255代表所有像素值除以255，0.0代表偏移值为0
    cv::Mat imageROI = image(cv::Rect(10,10,20,20)); // 选取图像的一个ROI区域
    // logo.copyTo(imageROI);   将logo图像（假如已经读取了）复制到imageROI区域      
    return 0;

}
