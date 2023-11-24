#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

// class HarrisDetector {
// public:
//     HarrisDetector()
//     : neighborhood(3),
//         aperture(3),
//         k(0.01),
//         maxStrength(0.0),
//         threshold(0.01),
//         nonMaxSize(3) {
//     // 创建用于非最大值抑制的内核
//     setLocalMaxWindowSize(nonMaxSize);
//     };

//     // 计算 Harris 角点强度
//     void detect(const cv::Mat& image) { 
//         // 计算 Harris 角点强度
//         cv::cornerHarris(image, cornerStrength, 
//         neighborhood,// 邻域尺寸
//         aperture, // 口径尺寸
//         k); 
//         // 计算内部阈值
//         cv::minMaxLoc(cornerStrength, 0, &maxStrength); 
//         // 检测局部最大值
//         cv::Mat dilated; // 临时图像
//         cv::dilate(cornerStrength, dilated, cv::Mat()); 
//         cv::compare(cornerStrength, dilated, localMax, cv::CMP_EQ); 
//     }
// private:
//     // 32 位浮点数型的角点强度图像
//     cv::Mat cornerStrength;
//     // 32 位浮点数型的阈值化角点图像
//     cv::Mat cornerTh;
//     // 局部最大值图像（内部）
//     cv::Mat localMax;
//     // 平滑导数的邻域尺寸
//     int neighborhood;
//     // 梯度计算的口径
//     int aperture;
//     // Harris 参数
//     double k;
//     // 阈值计算的最大强度
//     double maxStrength;
//     // 计算得到的阈值（内部）
//     double threshold;
//     // 非最大值抑制的邻域尺寸
//     int nonMaxSize;
//     // 非最大值抑制的内核
//     cv::Mat kernel;
// };

int main() {
    // 乱七八糟的，这一章节skip
    cv::Mat image = cv::imread("../workspace/corner.png", cv::IMREAD_GRAYSCALE);
    // 检测 Harris 角点
    cv::Mat cornerStrength;
    cv::cornerHarris(image,           // 输入图像
                     cornerStrength,  // 角点强度的图像
                     3,               // 邻域尺寸
                     3,               // 口径尺寸
                     0.01);           // Harris 参数
    // 由于角点强度值比较低都在1以下，所以保存下来的图都是黑色
    cv::imwrite("../workspace/cornerStrength.png", cornerStrength);
    // 对角点强度阈值化
    cv::Mat harrisCorners;
    double threshold = 0.0001;
    cv::threshold(cornerStrength, harrisCorners, threshold, 255,
                  cv::THRESH_BINARY);
    cv::imwrite("../workspace/harrisCorners.png", harrisCorners);
}
