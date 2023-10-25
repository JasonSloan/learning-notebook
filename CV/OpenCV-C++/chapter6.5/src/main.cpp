#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>


int main(){
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE);
    // 使用sobel算子进行边缘提取
    // ===============方式一===============
    // cv::Mat sobelX, sobelY;
    // cv::Sobel(image, sobelX, CV_16S, 1, 0, 3);
    // cv::Sobel(image, sobelY, CV_16S, 0, 1, 3);

    // // Convert back to CV_8U
    // cv::Mat sobelXAbs, sobelYAbs;
    // cv::convertScaleAbs(sobelX, sobelXAbs);
    // cv::convertScaleAbs(sobelY, sobelYAbs);

    // cv::Mat edgeImage;
    // cv::addWeighted(sobelXAbs, 0.5, sobelYAbs, 0.5, 0, edgeImage);
    // cv::imwrite("../workspace/sobel.png", edgeImage);
    // printf("Done!\n");


    // ===============方式二===============
    // 计算 Sobel 滤波器的范数
    cv::Mat sobelX, sobelY;
    cv::Sobel(image,sobelX,CV_16S,1,0); 
    cv::Sobel(image,sobelY,CV_16S,0,1); 
    cv::Mat sobel; 
    // 计算 L1 范数
    sobel= abs(sobelX)+abs(sobelY);
    // 找到 Sobel 最大值
    double sobmin, sobmax; 
    cv::minMaxLoc(sobel,&sobmin,&sobmax); 
    // 转换成 8 位图像
    // sobelImage = -alpha*sobel + 255 
    cv::Mat sobelImage; 
    sobel.convertTo(sobelImage,CV_8U,-255./sobmax,255); 
    cv::Mat sobelThresholded;
    cv::threshold(sobelImage, sobelThresholded, 
    220, 255, cv::THRESH_BINARY); 
    cv::imwrite("../workspace/sobel.png", sobelThresholded);
    printf("Done!\n");
}