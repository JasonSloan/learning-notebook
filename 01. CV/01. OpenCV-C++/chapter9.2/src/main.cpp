#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"


int main() {
    cv::Mat image = cv::imread("../workspace/corner.png", cv::IMREAD_GRAYSCALE);
    // 定义特征检测器
    cv::Ptr<cv::FeatureDetector> ptrDetector; // 泛型检测器指针
    ptrDetector= cv::FastFeatureDetector::create(80); 
//     // 检测关键点
//     ptrDetector->detect(image,keypoints1); 
//     ptrDetector->detect(image2,keypoints2);
// ;
    cv::imwrite("../workspace/harrisCorners.png", harrisCorners);
}
