#ifndef COLORHISTOGRAM_HPP
#define COLORHISTOGRAM_HPP

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"


class ColorHistogram { 
public: 
    ColorHistogram();
    void setSize(int size);     // 设置箱子数量
    cv::Mat getHistogram(const cv::Mat &image);     // 计算一维直方图
    cv::SparseMat getSparseHistogram(const cv::Mat &image);     // 计算三维直方图
    cv::Mat getHueHistogram(const cv::Mat &image, int minSaturation=0);     // 计算一维色调直方图
private: 
    int histSize[3]; // 每个维度的大小
    float hranges[2]; // 值的范围（三个维度用同一个值）
    const float* ranges[3]; // 每个维度的范围
    int channels[3]; // 需要处理的通道
 };

#endif