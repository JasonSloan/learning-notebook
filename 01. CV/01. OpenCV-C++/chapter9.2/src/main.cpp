#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main() {
    /*局部模板匹配实现思路:
    使用FAST检测器检测两幅图像中的关键点-->定义一个特定大小（例如11×11）的矩形，用于表示每个关键点周围的图像块-->
    将第一幅图像的关键点图像块与第二幅图像的关键点图像块一一进行比较(比较方法cv::matchTemplate)-->将最佳匹配的关键点对添加到标准化容器matches中
    */

    // 读取输入图像
    cv::Mat image1 = cv::imread("/root/study/opencv/workspace/1.jpg", 1);
    cv::Mat image2 = cv::imread("/root/study/opencv/workspace/2.jpg", 1);
    std::vector<cv::KeyPoint> keypoints1;  // 特征点向量
    std::vector<cv::KeyPoint> keypoints2;  // 特征点向量
    // 定义特征检测器
    cv::Ptr<cv::FeatureDetector> ptrDetector;  // 泛型检测器指针
    ptrDetector =                              // 这里选用 FAST 检测器
        cv::FastFeatureDetector::create(80);
    // 检测关键点
    ptrDetector->detect(image1, keypoints1);
    ptrDetector->detect(image2, keypoints2);
    // 这里采用了可以指向任何特征检测器的泛型指针类型cv::Ptr<cv::FeatureDetector>。
    // 然后定义一个特定大小（例如11×11）的矩形，用于表示每个关键点周围的图像块
    const int nsize = 11;                       // 邻域的尺寸
    cv::Rect neighborhood(0, 0, nsize, nsize);  // 11×11 x1y1wh(左上角+宽高)
    cv::Mat patch1;
    cv::Mat patch2;
    // 将一幅图像的关键点与另一幅图像的全部关键点进行比较。在第二幅图像中找出与第一幅图
    // 像中的每个关键点最相似的图像块。这个过程用两个嵌套循环实现
    cv::Mat result;
    std::vector<cv::DMatch> matches;
    // 针对图像一的全部关键点
    for (int i = 0; i < keypoints1.size(); i++) {
        // 定义图像块, 以关键点为中心, 11x11领域大小
        neighborhood.x = keypoints1[i].pt.x - nsize / 2;
        neighborhood.y = keypoints1[i].pt.y - nsize / 2;
        // 如果邻域超出图像范围，就继续处理下一个点
        if (neighborhood.x < 0 || neighborhood.y < 0 ||
            neighborhood.x + nsize >= image1.cols ||
            neighborhood.y + nsize >= image1.rows)
            continue;
        // 第一幅图像的块
        patch1 = image1(neighborhood);
        // 存放最匹配的值
        cv::DMatch bestMatch;
        // 针对第二幅图像的全部关键点
        for (int j = 0; j < keypoints2.size(); j++) {
            // 定义图像块, 以关键点为中心, 11x11领域大小
            neighborhood.x = keypoints2[j].pt.x - nsize / 2;
            neighborhood.y = keypoints2[j].pt.y - nsize / 2;
            // 如果邻域超出图像范围，就继续处理下一个点
            if (neighborhood.x < 0 || neighborhood.y < 0 ||
                neighborhood.x + nsize >= image2.cols ||
                neighborhood.y + nsize >= image2.rows)
                continue;
            // 第二幅图像的块
            patch2 = image2(neighborhood);
            // 匹配两个图像块
            cv::matchTemplate(patch1, patch2, result, cv::TM_SQDIFF);
            // 检查是否为最佳匹配, 如果是最佳匹配,
            // 就更新最佳匹配的queryIdx和trainIdx, bestMatch.distance初始值很大
            if (result.at<float>(0, 0) < bestMatch.distance) {
                bestMatch.distance = result.at<float>(0, 0);
                bestMatch.queryIdx = i;
                bestMatch.trainIdx = j;
            }
        }
        // 添加最佳匹配
        matches.push_back(bestMatch);
    }

    // 提取 25 个(可调)最佳匹配项, std::nth_element: 将容器中第n小的数放在第n个位置上, 第n个位置之前的数都比第n个位置小, 但是之前的数不一定是由小到大排列的
    std::nth_element(matches.begin(), matches.begin() + 25, matches.end());
    matches.erase(matches.begin() + 25, matches.end());
    // 画出匹配结果, 使用opencv自带的cv::drawMatches函数
    cv::Mat matchImage;
    cv::drawMatches(image1, keypoints1,          // 第一幅图像
                    image2, keypoints2,          // 第二幅图像
                    matches,                     // 匹配项的向量
                    matchImage,
                    cv::Scalar(255, 255, 255),   // 线条颜色
                    cv::Scalar(255, 255, 255));  // 点的颜色
    // 保存结果
    cv::imwrite("/root/study/opencv/workspace/matches.jpg", matchImage);
    return 0;
}
