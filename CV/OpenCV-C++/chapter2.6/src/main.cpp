#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>
#include <iostream>

void sharpen(const cv::Mat &image, cv::Mat &result){
    // 原理：sharpened_pixel= 5*current-left-right-up-down; 其实就是laplacian算子
    // 分配内存
    result.create(image.size(), image.type());
    int nchannels = image.channels();
    // 遍历每一行，不包括第一行和最后一行
    for(int j = 1; j < image.rows - 1; j++){
        const uchar* previous = image.ptr<const uchar>(j - 1);
        const uchar* current = image.ptr<const uchar>(j);
        const uchar* next = image.ptr<const uchar>(j + 1);
        uchar* output = result.ptr<uchar>(j);
        // 遍历每一列的每个像素，不包括第一列和最后一列
        for(int i = nchannels; i < (image.cols - 1) * nchannels; i++){
            // 因为计算后的像素值可能超过255或者小于0，所以要用saturate_cast函数
            *output++ = cv::saturate_cast<uchar>(5 * current[i] - current[i - 1] - current[i + 1] - previous[i] - next[i]);
        }
    }
    // 将图像的边界设置为0(.setTo方法可以一次性设置矩阵中的所有值)
    result.row(0).setTo(cv::Scalar(0));
    result.row(result.rows - 1).setTo(cv::Scalar(0));
    result.col(0).setTo(cv::Scalar(0));
    result.col(result.cols - 1).setTo(cv::Scalar(0));
}

void sharpen2D(const cv::Mat &image, cv::Mat &result) { 
    // 使用滤波器（卷积）实现锐化，原理同sharpen
    // 构造内核（所有入口都初始化为 0）
    cv::Mat kernel(3,3,CV_32F,cv::Scalar(0)); 
    // 对内核赋值
    kernel.at<float>(1,1)= 5.0; 
    kernel.at<float>(0,1)= -1.0; 
    kernel.at<float>(2,1)= -1.0; 
    kernel.at<float>(1,0)= -1.0; 
    kernel.at<float>(1,2)= -1.0; 
    // 对图像滤波
    cv::filter2D(image,result,image.depth(),kernel); 
} 

int main(){
    // 图像锐化
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_COLOR);
    cv::Mat result;
    sharpen(image, result);
    cv::imwrite("../workspace/dog_sharpen.png",result);
    return 0;

}