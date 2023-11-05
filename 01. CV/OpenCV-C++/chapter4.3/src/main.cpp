#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "histogram1d.hpp"

int main() {
    cv::Mat image = cv::imread("../workspace/dog.png", 0);
    // 查找表的用法（就是将图像中的每个像素值映射到另一个像素值，类似于pandas中的map）
    cv::Mat lut(1, 256, CV_8U); // 1 行，256 列，每个元素是 8 位无符号整数（代表这0-255的像素值）
    for (int i = 0; i < 256; ++i) {
        // 定义规则，规则代表每个像素值会变成原像素值的2倍
        lut.at<uchar>(0, i) = cv::saturate_cast<uchar>(2 * i); 
    }
    // 应用查找表
    cv::Mat adjustedImage;
    // 按照lut的规则，将image中的每个像素值映射到另一个像素值存储在adjustedImage
    cv::LUT(image, lut, adjustedImage);
    return 0;
}
