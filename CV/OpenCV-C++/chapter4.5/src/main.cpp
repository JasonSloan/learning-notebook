#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "histogram1d.hpp"

int main(){
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE);
    // Define the histogram parameters
    int histSize = 256; // Number of bins for the grayscale image
    float grayRange[] = {0, 256}; // Range for the grayscale values
    const float* ranges = {grayRange};
    int channels[] = {0}; // Channel to be used for histogram (grayscale)

    // Calculate the histogram of the grayscale image
    cv::MatND hist;
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, &histSize, &ranges, true, false);

    // Normalize the histogram
    cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);
    // ======== 以上是计算直方图的代码，和4.4节一模一样 ========
    // 计算直方图反向投影：也就是把图像上每个像素对应在直方图上的某个bin的概率值作为像素值
    cv::Mat backProjection;
    cv::calcBackProject(&image, 1, channels, hist, backProjection, &ranges, 1, true);
    }   