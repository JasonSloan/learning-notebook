#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>


int main(){
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE);
    // 闭运算：对图像先膨胀后腐蚀，去除黑色噪点
    cv::Mat element5(5,5,CV_8U,cv::Scalar(1)); 
    cv::Mat closed; 
    cv::morphologyEx(image,closed, // 输入和输出的图像 
    cv::MORPH_CLOSE, // 运算符
    element5); // 结构元素

    // 开运算：对图像先腐蚀后膨胀，去除白色噪点
    cv::Mat opened; 
    cv::morphologyEx(image, opened, cv::MORPH_OPEN, element5); 
}