#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

class Histogram1D { 
public: 
    Histogram1D();
    cv::Mat getHistogram(const cv::Mat &image);         // 计算一维直方图
    cv::Mat getHistogramImage(const cv::Mat &image, int zoom=1);    // 计算一维直方图图像
    static cv::Mat getImageOfHistogram (const cv::Mat &hist, int zoom);     // 计算一维直方图图像（静态方法）
    static cv::Mat applyLookUp(const cv::Mat& image, const cv::Mat& lookup);    // 应用查找表（类似于pandas中的map）
private: 
    int histSize[1]; // 直方图中箱子的数量
    float hranges[2]; // 值范围
    const float* ranges[1]; // 值范围的指针
    int channels[1]; // 要检查的通道数量
};

