#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "colorHistogram.hpp"

ColorHistogram::ColorHistogram() { 
    // 准备用于彩色图像的默认参数
    // 每个维度的大小和范围是相等的
    histSize[0]= histSize[1]= histSize[2]= 256; 
    hranges[0]= 0.0; // BGR 范围为 0~256 
    hranges[1]= 256.0; 
    ranges[0]= hranges; // 这个类中
    ranges[1]= hranges; // 所有通道的范围都相等
    ranges[2]= hranges; 
    channels[0]= 0; // 三个通道：B 
    channels[1]= 1; // G 
    channels[2]= 2; // R 
}

void ColorHistogram::setSize(int size) { 
    // 设置箱子数量
    histSize[0]= histSize[1]= histSize[2]= size; 
}

// 计算直方图
cv::Mat ColorHistogram::getHistogram(const cv::Mat &image) { 
    cv::Mat hist; 
    // 计算直方图
    cv::calcHist(&image, 1, // 单幅图像的直方图
    channels, // 用到的通道
    cv::Mat(), // 不使用掩码
    hist, // 得到的直方图
    3, // 这是一个三维直方图
    histSize, // 箱子数量
    ranges // 像素值的范围
    ); 
    return hist; 
} 

// 计算多维直方图
cv::SparseMat ColorHistogram::getSparseHistogram(const cv::Mat &image) { 
    cv::SparseMat hist(3, // 维数
    histSize, // 每个维度的大小
    CV_32F); 
    // 计算直方图
    cv::calcHist(&image, 1, // 单幅图像的直方图
    channels, // 用到的通道
    cv::Mat(), // 不使用掩码
    hist, // 得到的直方图
    3, // 这是三维直方图
    histSize, // 箱子数量
    ranges // 像素值的范围
    ); 
    return hist; 
} 


// 计算一维色调直方图
// BGR 的原图转换成 HSV
// 忽略低饱和度的像素
cv::Mat ColorHistogram::getHueHistogram(const cv::Mat &image, int minSaturation) { 
    cv::Mat hist; 
    // 转换成 HSV 色彩空间
    cv::Mat hsv; 
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV); 
    // 掩码（可能用到，也可能用不到）
    cv::Mat mask; 
    // 根据需要创建掩码
    if (minSaturation>0) { 
    // 将 3 个通道分割进 3 幅图像 
    std::vector<cv::Mat> v; 
    cv::split(hsv,v); 
    // 屏蔽低饱和度的像素
    cv::threshold(v[1],mask,minSaturation, 
    255, cv::THRESH_BINARY); 
    } 
    // 准备一维色调直方图的参数
    hranges[0]= 0.0; // 范围为 0~180 
    hranges[1]= 180.0; 
    channels[0]= 0; // 色调通道
    // 计算直方图
    cv::calcHist(&hsv, 1, // 只有一幅图像的直方图
    channels, // 用到的通道
    mask, // 二值掩码
    hist, // 生成的直方图
    1, // 这是一维直方图
    histSize, // 箱子数量
    ranges // 像素值范围
    ); 
    return hist; 
} 

