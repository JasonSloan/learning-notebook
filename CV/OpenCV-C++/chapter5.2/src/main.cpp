#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>


int main(){
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE);
    // 腐蚀图像，黑变大
    // 采用默认的 3×3 结构元素
    cv::Mat eroded; // 目标图像
    cv::erode(image,eroded,cv::Mat());

    // 膨胀图像，白变大
    cv::Mat dilated; // 目标图像
    cv::dilate(image,dilated,cv::Mat());
    
    // 用更大的元素腐蚀图像
    cv::Mat element(7,7,CV_8U,cv::Scalar(1)); 
    // 用这个结构元素腐蚀图像
    cv::erode(image,eroded,element);
    return 0;
}