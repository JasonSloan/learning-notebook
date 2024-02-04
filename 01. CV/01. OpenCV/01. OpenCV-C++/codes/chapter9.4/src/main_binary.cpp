#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main() {
    /*描述和匹配兴趣点: 二值描述子
    二值描述子主要是ORB和BRISK*/

    // 读取输入图像
    cv::Mat image1 = cv::imread("/root/study/opencv/workspace/1.jpg", 1);
    cv::Mat image2 = cv::imread("/root/study/opencv/workspace/2.jpg", 1);
    // 定义关键点容器和描述子
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors1;
    cv::Mat descriptors2;
    // 定义特征检测器/描述子// 大约 60 个特征点
    cv::Ptr<cv::Feature2D> feature = cv::ORB::create(60);
    // 检测并描述关键点
    // 检测 ORB 特征
    feature->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    feature->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
    // 构建匹配器
    cv::BFMatcher matcher(cv::NORM_HAMMING); // 二值描述子一律使用 Hamming 规范
    // 匹配两幅图像的描述子
    std::vector<cv::DMatch> matches; 
    matcher.match(descriptors1, descriptors2, matches);

    cv::Mat imageMatches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, imageMatches);
    // 保存
    cv::imwrite("/root/study/opencv/workspace/matches.jpg", imageMatches);

    return 0;
}
