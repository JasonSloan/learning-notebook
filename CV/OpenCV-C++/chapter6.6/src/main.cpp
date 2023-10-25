#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>


int main(){
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE);
    cv::Mat laplace;
    int aperture = 3;
    cv::Laplacian(image, laplace, CV_32F, aperture);
    double lapmin, lapmax;
    cv::minMaxLoc(laplace, &lapmin, &lapmax);
    printf("lapmin:%f, lapmax:%f\n",lapmin, lapmax);
    double maximum = std::max(-lapmin, lapmax);
    printf("std::max(-lapmin, lapmax):%f\n", maximum);
    double scale = 127 / std::max(-lapmin, lapmax);    // laplace中任何一个数乘以scale后的值域[-127, 127]
    printf("The scale %f\n", scale);
    cv::Mat laplaceImage;
    laplace.convertTo(laplaceImage, CV_8U, scale, 128);     // laplace中任何一个数乘以scale+128后的值域[1, 255]
    cv::imwrite("../workspace/laplace.png", laplaceImage);
    printf("Done!\n");
}
