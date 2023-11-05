#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>

int main(){
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_COLOR);
    cv::cvtColor(image,image,cv::COLOR_BGR2RGB);
    // std::vector<cv::Mat> channels; 
    cv::Mat channels[3];
    cv::split(image, channels);
}