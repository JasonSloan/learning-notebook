#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>


int main(){
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE);
    cv::Mat result;
    // 中值滤波（最适合消除椒盐噪声）
    cv::medianBlur(image, result, 5); 
    printf("Done!\n");
}