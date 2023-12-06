#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main() {
    /*描述和匹配兴趣点: 去除冗余匹配点
    方式二: 匹配差值的阈值化
    还有一种更加简单的策略，就是把描述子之间差值太大的匹配项排除。实现此功能的是
    cv::DescriptorMatcher 类的 radiusMatch 方法*/

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

    // 构造匹配器, cv::NORM_L2: 匹配度量方式  true: 使用交叉检查
    cv::BFMatcher matcher(cv::NORM_L2);
    // 匹配两幅图像的描述子
    // 为每个关键点找出两个最佳匹配项
    std::vector<std::vector<cv::DMatch>> matches;
    // 指定范围的匹配
    // 两个描述子之间的最大允许差值
    float maxDist = 0.4;
    matcher.radiusMatch(descriptors1, descriptors2, matches, maxDist);
    
    cv::Mat imageMatches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches,
                    imageMatches);
    // 保存
    cv::imwrite("/root/study/opencv/workspace/matches.jpg", imageMatches);

    return 0;
}
