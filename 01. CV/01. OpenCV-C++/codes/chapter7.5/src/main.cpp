#include <stdio.h>
#include <cmath>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

int main() {
    // 像素强度有明显变化的点叫边缘
    // 连续的像素强度一致的点叫轮廓
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE);
    cv::Mat binary;
    cv::threshold(image, binary, 114, 255, cv::THRESH_BINARY);
    // 用于存储轮廓的向量
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary,
                     contours,                // 存储轮廓的向量
                     cv::RETR_EXTERNAL,       // 检索外部轮廓
                     cv::CHAIN_APPROX_NONE);  // 每个轮廓的全部像素
    printf("Find %ld contours!\n", contours.size());

    // 删除太短或太长的轮廓
    int cmin = 10;    // 最小轮廓长度
    int cmax = 500;  // 最大轮廓长度
    std::vector<std::vector<cv::Point>>::iterator itc = contours.begin();
    // 针对所有轮廓
    while (itc != contours.end()) {
        // 验证轮廓大小
        if (itc->size() < cmin || itc->size() > cmax)
            itc = contours.erase(itc);
        else
            ++itc;
    }
    printf("After filter, %ld contours remained!\n", contours.size());

    // 在白色图像上画黑色轮廓
    cv::Mat result(image.size(), CV_8U, cv::Scalar(255));
    cv::drawContours(result, contours,
                     -1,  // 画全部轮廓
                     0,   // 用黑色画
                     2);  // 宽度为 2
    cv::imwrite("../workspace/contours.png", result);
    printf("Done!\n");
}
