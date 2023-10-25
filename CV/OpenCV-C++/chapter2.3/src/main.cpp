#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>
#include <iostream>

int main(){
    cv::Mat image;
    image = cv::imread("../workspace/dog.png", cv::IMREAD_COLOR);
    cv::Vec3b vec = image.at<cv::Vec3b>(0,0); // 读取图像的第一个像素点
    printf("B: %d, G: %d, R: %d\n", vec[0], vec[1], vec[2]);
    cv::Mat gray;
    gray = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE);
    uchar intensity = gray.at<uchar>(0,0); // 读取图像的第一个像素点
    printf("Gray: %d\n", intensity);
    std::cout << "The length of each rows:" << image.step << std::endl; // 考虑内存对齐（填充）的每一行的字节数
    std::cout << "The length of each rows:" <<image.cols * image.elemSize() << std::endl;   // 没有考虑内存对齐的每一行的字节数
    uchar* data = image.data;
    data += image.step; // 一行一行移动指针
    // 注意区分image.at和image.ptr取像素值的区别
    // 遍历图像上每一个像素点，方式一，指定行号和列号，取出每一个像素点
    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            cv::Vec3b vec = image.at<cv::Vec3b>(i,j);
            printf("B: %d, G: %d, R: %d\n", vec[0], vec[1], vec[2]);
        }
    }
    // 遍历图像上每一个像素点，方式二，指定行号，取出每一行的首地址，然后遍历每一行的每一个像素点
    /*BGR BGR BGR BGR BGR BGR BGR BGR BGR BGR BGR BGR......
     *BGR BGR BGR BGR BGR BGR BGR BGR BGR BGR BGR BGR......
     *BGR BGR BGR BGR BGR BGR BGR BGR BGR BGR BGR BGR......
     *BGR BGR BGR BGR BGR BGR BGR BGR BGR BGR BGR BGR......*/
    int rows = image.rows;
    int cols = image.cols * image.channels();    // 每一行像素值的个数
    for(int i=0; i< rows;i++){
        uchar* row_ptr = image.ptr<uchar>(i);
        for(int j=0; j < cols; j++){
            printf("%d ", row_ptr[j]);
        }
    }
}