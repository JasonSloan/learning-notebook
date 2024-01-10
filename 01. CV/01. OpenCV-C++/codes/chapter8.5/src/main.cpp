#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main() {
    /* BRISK和ORB兴趣点检测特点: 基于FAST检测器的多尺度版本, 和SURF以及SIFT一样, 具有方向不变性, 尺度不变性*/
    // 读取输入图像
    cv::Mat image = cv::imread("/root/study/opencv/workspace/1.jpg", 1);
    // ==================== BRISK 特征检测 ====================
    std::vector<cv::KeyPoint> keypoints;
    // 构造 BRISK 特征检测器对象
    cv::Ptr<cv::BRISK> ptrBRISK = cv::BRISK::create();
    // 检测关键点
    ptrBRISK->detect(image, keypoints);

    // ==================== ORB 特征检测 ====================
    // 构造 ORB 特征检测器对象
    cv::Ptr<cv::ORB> ptrORB = cv::ORB::create(75,  // 关键点的总数
                                              1.2,  // 图层之间的缩放因子
                                              8);  // 金字塔的图层数量
    // 检测关键点
    ptrORB->detect(image, keypoints);

    return 0;
}
