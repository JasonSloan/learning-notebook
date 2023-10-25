#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "histogram1d.hpp"

int main(){
    // // 统计图像的直方图
    // // ====================方式一====================
    // cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE); // Load a grayscale image
    // int histSize = 256; // Number of bins
    // float range[] = {0, 256}; // Range of values
    // const float* histRange = {range};   // 因为是灰度图，只有一个通道

    // cv::Mat hist; // Output histogram
    // // 1：图像个数，0：通道索引，Mat()：掩码，hist：输出的直方图，1：这是1维的直方图，histSize：直方图bins的个数，histRange：直方图的范围
    // cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    // printf("The size of the hist is %d\n",hist.rows);

    // // Normalize the histogram for visualization
    // // cv::normalize(src, dst, alpha, beta, norm_type)
    // // alpha是归一化后的最小值，beta是归一化后的最大值，norm_type是归一化的类型 
    // cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

    // // Create an empty image to draw the histogram
    // int histWidth = 512;
    // int histHeight = 400;
    // cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    // // Draw the histogram bars
    // int binWidth = cvRound(static_cast<double>(histWidth) / histSize);
    // for (int i = 0; i < histSize; i++) {
    //     cv::rectangle(histImage,
    //                   cv::Point(i * binWidth, histHeight),  // bin的左下角
    //                   cv::Point((i + 1) * binWidth, histHeight - cvRound(hist.at<float>(i))), 
    //                   cv::Scalar(0, 0, 0),
    //                   -1);
    // }

    // // Display the histogram
    // cv::imwrite("../workspace/dog_hist.png",histImage);

    // ====================方式二====================
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE); // Load a grayscale image
    Histogram1D h; // Instantiate a histogram object
    cv::Mat histo;
    histo = h.getHistogram(image); // Compute the histogram
    cv::Mat histImg = h.getHistogramImage(image); // Display the histogram
    cv::imwrite("../workspace/dog_hist.png",histImg);
}