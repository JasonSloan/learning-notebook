#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main() {
    /*描述和匹配兴趣点: 去除冗余匹配点
    方式一: 交叉检查(效果一般不好)
    有一种简单的方法可以验证得到的匹配项，即重新进行同一个匹配过程(第一次匹配是从计算image1的兴趣点与image2的兴趣点的距离进行的匹配)，
    但在第二次匹配时(第二次匹配是从计算image2的兴趣点与image1的兴趣点的距离进行的匹配)，将第二幅图像的每个关键点逐个与第一幅图像的全部关键点进行比较。
    只有在两个方向都匹配了同一对关键点（即两个关键点互为最佳匹配）时，才认为是一个有效的匹配项。函数
    cv::BFMatcher提供了一个选项来使用这个策略。 把有关标志设置为
    true，函数就会对匹配进行双向的交叉检查*/

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
    cv::BFMatcher matcher(cv::NORM_L2, true);
    // 匹配两幅图像的描述子
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    cv::Mat imageMatches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches,
                    imageMatches);
    // 保存
    cv::imwrite("/root/study/opencv/workspace/matches.jpg", imageMatches);

    return 0;
}
