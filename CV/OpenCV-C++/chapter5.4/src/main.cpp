#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>


int main(){
    /*
    理解形态学运算在灰度图像上的效果有一个好办法，就是把图像看作是一个拓扑地貌，不同
    的灰度级别代表不同的高度（或海拔）。基于这种观点，明亮的区域代表高山，黑暗的区域代表
    108 第 5 章 用形态学运算变换图像
    深谷；边缘相当于黑暗和明亮像素之间的快速过渡，因此可以比作陡峭的悬崖。腐蚀这种地形的
    最终结果是：每个像素被替换成特定邻域内的最小值，从而降低它的高度。结果是悬崖“缩小”，
    山谷“扩大”。膨胀的效果刚好相反，即悬崖“扩大”，山谷“缩小”。但不管哪种情况，平地（即
    强度值固定的区域）都会相对保持不变。
    根据这个结论，可以得到一种检测图像边缘（或悬崖）的简单方法，即通过计算膨胀后的图
    像与腐蚀后的图像之间的的差距得到边缘。因为这两种转换后图像的差别主要在边缘地带，所以
    相减后会突出边缘。在 cv::morphologyEx 函数中输入 cv::MORPH_GRADIENT 参数，即可实
    现此功能。显然，结构元素越大，检测到的边缘就越宽。这种边缘检测运算称为 Beucher 梯度
    （下一章将详细讨论图像梯度的概念）。注意还有两种简单的方法能得到类似结果，即用膨胀后的
    图像减去原始图像，或者用原始图像减去腐蚀后的图像，那样得到的边缘会更窄。
    */
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE);
    cv::Mat result; 
    cv::morphologyEx(image, result, 
    cv::MORPH_GRADIENT, cv::Mat()); 

    /*
    顶帽运算也基于图像比对，它使用了开启和闭合运算。因为灰度图像进行形态学开启运算时
    会先对图像进行腐蚀，局部的尖锐部分会被消除，其他部分则将保留下来。因此，原始图像和经
    过开启运算的图像的比对结果就是局部的尖锐部分。这些尖锐部分就是我们需要提取的前景物
    体。对于本书的照片来说，前景物体就是页面上的文字。因为书本为白底黑字，所以我们采用它
    的互补运算，即黑帽算法。它将对图像做闭合运算，然后从得到的结果中减去原始图像。这里采
    用 7×7 的结构元素，它足够大了，能确保移除文字
    */
    // 使用 7×7 结构元素做黑帽变换
    cv::Mat element7(7, 7, CV_8U, cv::Scalar(1)); 
    cv::morphologyEx(image, result, cv::MORPH_BLACKHAT, element7);
    printf("Done!\n");
}