#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "histogram1d.hpp"


 // 准备一维直方图的默认参数
Histogram1D::Histogram1D() {
    histSize[0]= 256; // 256 个箱子
    hranges[0]= 0.0; // 从 0 开始（含）
    hranges[1]= 256.0; // 到 256（不含）
    ranges[0]= hranges; 
    channels[0]= 0; // 先关注通道 0
} 

// 计算一维直方图
cv::Mat Histogram1D::getHistogram(const cv::Mat &image) { 
    cv::Mat hist; 
    // 用 calcHist 函数计算一维直方图
    cv::calcHist(&image, 1, // 仅为一幅图像的直方图
    channels, // 使用的通道
    cv::Mat(), // 不使用掩码
    hist, // 作为结果的直方图
    1, // 这是一维的直方图
    histSize, // 箱子数量
    ranges // 像素值的范围
    );
    return hist; 
}

// 计算一维直方图，并返回它的图像
cv::Mat Histogram1D::getHistogramImage(const cv::Mat &image, int zoom) { 
    // 先计算直方图
    cv::Mat hist= getHistogram(image); 
    // 创建图像
    return getImageOfHistogram(hist, zoom); 
} 

// 创建一个表示直方图的图像（静态方法）
cv::Mat Histogram1D::getImageOfHistogram (const cv::Mat &hist, int zoom) { 
    // 取得箱子值的最大值和最小值
    double maxVal = 0; 
    double minVal = 0; 
    cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0); 
    // 取得直方图的大小
    int histSize = hist.rows; 
    // 用于显示直方图的方形图像
    cv::Mat histImg(histSize*zoom, histSize*zoom, 
    CV_8U, cv::Scalar(255)); 
    // 设置最高点为 90%（即图像高度）的箱子个数
    int hpt = static_cast<int>(0.9*histSize); 
    // 为每个箱子画垂直线
    for (int h = 0; h < histSize; h++) { 
        float binVal = hist.at<float>(h); 
        if (binVal>0) { 
            int intensity = static_cast<int>(binVal*hpt / maxVal); 
            cv::line(histImg, cv::Point(h*zoom, histSize*zoom), 
            cv::Point(h*zoom, (histSize - intensity)*zoom), 
            cv::Scalar(0), zoom); 
        } 
    } 
    return histImg; 
} 


// 对图像应用查找表
cv::Mat Histogram1D::applyLookUp(const cv::Mat& image, // 输入图像
    const cv::Mat& lookup) {// uchar 类型的 1×256 数组
    // 输出图像
    cv::Mat result; 
    // 应用查找表
    cv::LUT(image,lookup,result); 
    return result; 
} 