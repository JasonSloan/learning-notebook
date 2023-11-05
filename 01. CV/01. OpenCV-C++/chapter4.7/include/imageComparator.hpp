#ifndef IMAGE_COMPARATOR_HPP
#define IMAGE_COMPARATOR_HPP
#include "opencv2/core.hpp"
#include "colorHistogram.hpp"


class ImageComparator { 
 public: 
    ImageComparator(); // 默认箱子数量为 8
    // 设置并计算基准图像的直方图
    void setReferenceImage(const cv::Mat& image);
    double compare(const cv::Mat& image);
 private: 
    cv::Mat refH; // 基准直方图
    cv::Mat inputH; // 输入图像的直方图
    ColorHistogram hist; // 生成直方图
    int nBins; // 每个颜色通道使用的箱子数量
};

#endif // IMAGE_COMPARATOR_HPP