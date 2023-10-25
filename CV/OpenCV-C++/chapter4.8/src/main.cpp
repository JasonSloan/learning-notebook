#include "colorHistogram.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

int main() {
    // ==================== 积分图像 ====================
    cv::Mat image = cv::imread("../workspace/dog.png", 1);
    // 定义图像的 ROI（这里为骑自行车的女孩）
    int xo = 1, yo = 1;
    int width = 10, height = 20;
    cv::Mat roi(image, cv::Rect(xo, yo, width, height));
    // 计算累加值
    // 返回一个多通道图像下的 Scalar 数值
    cv::Scalar sum = cv::sum(roi);  // 这种方式速度很慢

    // 使用积分图像的方式只需要一次扫描全图的计算量，后续使用时只需要一次加减运算即可得到任意区域的累加值。
    // 计算积分图像
    cv::Mat integralImage;
    cv::integral(image, integralImage, CV_32S);
    // 此时直接用1次加法，2次减法直接就能得到任意区域的像素的累加值，速度极快
    int sumInt = integralImage.at<int>(yo + height, xo + width) -
                 integralImage.at<int>(yo + height, xo) -
                 integralImage.at<int>(yo, xo + width) +
                 integralImage.at<int>(yo, xo);
    // ==================== 自适应阈值二值化 ====================
    /*原理：对于一张图像，如果我们将图像分成若干个小区域，那么每个小区域的像素值的分布情况可能是不一样的，
     * 有的区域的像素值可能比较集中，有的区域的像素值可能比较分散。如果我们对每个小区域分别计算一个阈值，
     * 那么这个阈值就可以根据这个小区域的像素值的分布情况来确定，这样就可以得到一系列的阈值，
     * 然后利用这些阈值一个局部一个局部的做二值化，这样每个局部只跟自己相关。
     */
    cv::Mat image_ = cv::imread("../workspace/3.jpg", 0);
    cv::Mat binaryAdaptive;
    int blockSize = 21;  // 每个小区域的大小
    int threshold = 10;  // 计算阈值，这里是平均值减去这个偏移量
    cv::adaptiveThreshold(image_,                      // 输入图像
                          binaryAdaptive,              // 输出二值图像
                          255,                         // 输出的最大值
                          cv::ADAPTIVE_THRESH_MEAN_C,  // 方法
                          cv::THRESH_BINARY,           // 阈值类型
                          blockSize,                   // 块的大小
                          threshold);                  // 使用的阈值
    cv::imwrite("../workspace/3_1.jpg", binaryAdaptive);
    printf("自适应阈值二值化完成\n");
};