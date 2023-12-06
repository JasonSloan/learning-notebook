#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main() {
    /* FAST兴趣点检测特点: 速度非常快, 具有方向不变性的特点, 但是不具有尺度不变性
    FAST 对角点的定义基于候选特征点周围的图像强度值。以某个点为中心做一个圆，根据圆上的像素值判断该
    点是否为关键点。如果存在这样一段圆弧，它的连续长度超过周长的 3/4，并且它上面所有像素
    的强度值都与圆心的强度值明显不同（全部更暗或更亮），那么就认定这是一个关键点。*/
    // 读取输入图像
    cv::Mat image = cv::imread("/root/study/opencv/workspace/1.jpg", 1);
    // 关键点的向量
    std::vector<cv::KeyPoint> keypoints;
    // FAST 特征检测器, 阈值为 40
    /* 一个点与圆心强度值的差距必须达到一个指定的值，才能被认为是明显更暗或更亮；这个值
    就是创建检测器实例时指定的阈值参数。这个阈值越大，检测到的角点数量就越少。*/
    cv::Ptr<cv::FastFeatureDetector> ptrFAST =
        cv::FastFeatureDetector::create(40);
    // 检测关键点
    ptrFAST->detect(image, keypoints);
    // 画出关键点
    cv::drawKeypoints(image,                      // 原始图像
                      keypoints,                  // 关键点的向量
                      image,                      // 输出图像
                      cv::Scalar(255, 255, 255),  // 关键点的颜色
                      cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);  // 画图标志
    // 保存输出图像
    cv::imwrite("/root/study/opencv/workspace/1_fast.jpg", image);
    return 0;
}