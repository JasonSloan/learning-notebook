
#include <stdio.h>
#include <cmath>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

int main() {
    cv::Mat image = cv::imread("../workspace/3.jpg", cv::IMREAD_COLOR);
    cv::Mat data(image.size(), image.type());
    for(int i=0;i<image.rows;i++){
        for(int j=0;j<image.cols;j++){
            cv::Vec3b values = image.at<cv::Vec3b>(i, j);
            data[]
        }
        
    }
    
    // cv::Mat binary;
    // // 背景必须是黑色
    // cv::threshold(image, binary, 110, 255, cv::THRESH_BINARY_INV);
    // // 用于存储轮廓的向量
    // std::vector<std::vector<cv::Point>> contours;
    // cv::findContours(binary,
    //                  contours,                // 存储轮廓的向量
    //                  cv::RETR_EXTERNAL,       // 检索外部轮廓
    //                  cv::CHAIN_APPROX_NONE);  // 每个轮廓的全部像素
    // // 测试边界框
    // // 画最小覆盖矩形(类似于目标检测中画矩形框)
    // cv::Mat result1(image.size(), CV_8U, cv::Scalar(114));
    // for (int i = 0; i < contours.size(); i++) {
    //     cv::Rect r0 = cv::boundingRect(contours[i]);
    //     cv::rectangle(result1, r0, 0, 2);
    // }
    // cv::imwrite("../workspace/rect_contours.png", result1);

    // // 最小覆盖圆的情况也类似，将它用于右上角的区域：
    // // 测试覆盖圆
    // float radius;
    // cv::Point2f center;
    // cv::Mat result2(image.size(), CV_8U, cv::Scalar(114));
    // for (int i = 0; i < contours.size(); i++) {
    //     cv::minEnclosingCircle(contours[i], center, radius);
    //     cv::circle(result2, center, static_cast<int>(radius), cv::Scalar(0), 2);
    // }
    // cv::imwrite("../workspace/circle_contours.png", result2);

    // // 最小覆盖多边形
    // // 测试多边形逼近
    // std::vector<cv::Point> poly;
    // cv::approxPolyDP(contours[2], poly, 5, true);
    // cv::Mat result3(image.size(), CV_8U, cv::Scalar(114));
    // // 画多边形
    // for (int i = 0; i < contours.size(); i++) {
    //     cv::approxPolyDP(contours[i], poly, 5, true);
    //     cv::polylines(result3, poly, true, 0, 2);
    // }
    // cv::imwrite("../workspace/poly_contours.png", result3);

    // // 凸包缺陷检测参考：https://blog.csdn.net/wenhao_ir/article/details/51802882?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%87%B8%E5%8C%85%E7%BC%BA%E9%99%B7&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-51802882.142^v93^chatsearchT3_2&spm=1018.2226.3001.4187

    printf("Done!\n");
}
