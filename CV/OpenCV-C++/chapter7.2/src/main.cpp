#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>


int main(){
    // canny算法实现step by step(需要梯子)：https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE);
    cv::Mat contours; 
    cv::Canny(image, // 灰度图像
    contours, // 输出轮廓
    125, // 低阈值
    350); // 高阈值
    cv::imwrite("../workspace/canny.png", contours);
    printf("Done!\n");
}