#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "imageComparator.hpp"


ImageComparator::ImageComparator() :nBins(8){}; // 默认箱子数量为 8

// 设置并计算基准图像的直方图
void ImageComparator::setReferenceImage(const cv::Mat& image) { 
    hist.setSize(nBins); 
    refH = hist.getHistogram(image); 
} 

// 用 BGR 直方图比较图像
double ImageComparator::compare(const cv::Mat& image) { 
    inputH = hist.getHistogram(image); 
    // 用交叉法比较直方图, 越大越相似。这个类最主要的就是这个方法：接收两个参数，一个参数是基准直方图，一个参数是输入直方图，返回值是两个直方图的相似度。
    return cv::compareHist(refH, inputH, cv::HISTCMP_INTERSECT); 
} 