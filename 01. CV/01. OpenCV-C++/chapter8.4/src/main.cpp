#include "opencv2/opencv.hpp"
#include <iostream>


using namespace cv;
using namespace std;

int main() {
    /* SURF和SIFT兴趣点检测特点: 具有方向不变性, 尺度不变性; SIFT比SURF更加稳定, 但是SURF更加快速;
    尺度不变性:
    不仅在任何尺度下拍摄的物体都能检测到一致的关键点，而且每个被检测的特征点都对应一个尺
    度因子。理想情况下，对于两幅图像中不同尺度的同一个物体点，计算得到的两个尺度因子之间
    的比率应该等于图像尺度的比率。近几年，人们提出了多种尺度不变特征，本节将介绍其中的一
    种：SURF 特征，它的全称为加速稳健特征（Speeded Up Robust
    Feature）。我们将会看到，它们不仅是尺度不变特征，而且是具有较高计算效率的特征。
    */
    // 读取输入图像
    cv::Mat image = cv::imread("/root/study/opencv/workspace/1.jpg", 1);
    std::vector<cv::KeyPoint> keypoints;
    // ==================== SURF 特征检测 ====================
    // // 创建 SURF 特征检测器对象
    // cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> ptrSURF =
    //     cv::xfeatures2d::SurfFeatureDetector::create(2000.0);
    // // 检测关键点
    // ptrSURF->detect(image, keypoints);

    // ==================== SIFT 特征检测 ====================
    // 构建 SIFT 特征检测器实例
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create(250);
    detector->detect(image, keypoints);

    return 0;
}

