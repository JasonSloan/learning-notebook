#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main() {
    /*描述和匹配兴趣点: 去除冗余匹配点
    方式二: 比率检查
    因为场景中有很多相似的物体，一个关键点可以与多个其他关键点匹配。所以错误的匹配项非常多.
    为此我们需要为每个关键点找到两个最佳的匹配项. 可以用 cv::Descriptor Matcher
    类 的 knnMatch 方法实现这个功能
    下一步是排除与第二个匹配项非常接近的全部最佳匹配项。因为 knnMatch 生成了一个
    std::vector 类型（此向量的长度为 k）的 std::vector
    类，所以这一步的具体做法是循环遍
    历每个关键点匹配项，然后执行比率检验法，即计算排名第二的匹配项与排名第一的匹配项的差
    值之比（如果两个最佳匹配项相等，那么比率为
    1）。比率值较高的匹配项将作为模糊匹配项， 从结果中被排除掉*/

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
    matcher.knnMatch(descriptors1, descriptors2, matches,
                     2);  // 找出 k 个最佳匹配项
    
    // 执行比率检验法, ratio越低, 匹配到的点越少
    double ratio = 0.7;
    std::vector<std::vector<cv::DMatch>>::iterator it;
    std::vector<cv::DMatch> newMatches;
    for (it = matches.begin(); it != matches.end(); ++it) {
        // 第一个最佳匹配项/第二个最佳匹配项
        if ((*it)[0].distance / (*it)[1].distance < ratio) {
            // 这个匹配项可以接受
            newMatches.push_back((*it)[0]);
        }
    }
    // newMatches 是新的匹配项集合

    cv::Mat imageMatches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, newMatches,
                    imageMatches);
    // 保存
    cv::imwrite("/root/study/opencv/workspace/matches.jpg", imageMatches);

    return 0;
}
