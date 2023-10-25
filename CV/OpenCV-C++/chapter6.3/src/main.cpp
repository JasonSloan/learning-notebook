#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>


int main(){
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE);
    cv::Mat resizedImage; // 用于存储缩放后的图像
    cv::resize(image, resizedImage, 
    cv::Size(image.cols/4,image.rows/4)); // 行和列均缩小为原来的 1/4 
    // 你也可以指定缩放比例。在参数中提供一个空的图像实例，然后提供缩放比例：
    cv::resize(image, resizedImage, 
    cv::Size(), 1.0/4.0, 1.0/4.0); // 缩小为原来的 1/4
    // cv::INTER_NEAREST：使用双线性插值的方式对图像进行缩放，resize默认的插值方式就是这个，所以可以不用指定
    cv::resize(image, resizedImage, cv::Size(), 3, 3, cv::INTER_NEAREST); 
    printf("Done!\n");
}