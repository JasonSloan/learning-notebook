#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main() {
    /*描述和匹配兴趣点: 使用SIFT检测并描述兴趣点. 检测器和描述子可以相同,
    检测器和描述子也可以任意搭配 SURF描述子的默认尺寸是 64，而 SIFT 的默认尺寸是
    128*/

    // 读取输入图像
    cv::Mat image1 = cv::imread("/root/study/opencv/workspace/1.jpg", 1);
    cv::Mat image2 = cv::imread("/root/study/opencv/workspace/2.jpg", 1);
    // 定义关键点的容器
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    // 定义特征检测器
    cv::Ptr<cv::Feature2D> ptrFeature2D = cv::SIFT::create(2000.0);
    // 检测关键点
    ptrFeature2D->detect(image1, keypoints1);
    ptrFeature2D->detect(image2, keypoints2);
    // 提取描述子
    cv::Mat descriptors1;
    cv::Mat descriptors2;
    ptrFeature2D->compute(image1, keypoints1, descriptors1);
    ptrFeature2D->compute(image2, keypoints2, descriptors2);

    // 构造匹配器
    cv::BFMatcher matcher(cv::NORM_L2);
    // 匹配两幅图像的描述子
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    cv::Mat imageMatches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, imageMatches);
    // 保存
    cv::imwrite("/root/study/opencv/workspace/matches.jpg", imageMatches);

    return 0;
}
