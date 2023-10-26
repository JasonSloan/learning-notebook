#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>

int main(){
    // 失败？未找到原因
    // 使用GrabCut算法进行前景图像分割
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_COLOR);
    cv::Mat result; // 分割结果，四种可能值，0：背景，1：前景，2：可能是背景，3：可能是前景
    cv::Mat bgModel,fgModel;    // 模型内部用
    cv::Rect rect(0,0,image.cols,image.rows); // 指定前景所在的矩形区域（相当于目标检测的框）
    cv::grabCut(image,result,rect,bgModel,fgModel,5,cv::GC_INIT_WITH_RECT); // 5：迭代5次；GC_INIT_WITH_RECT:使用矩形区域
    // cv::compare(src1, src2, dst, cmp_op)，比较result和GC_PR_FGD，相等的像素点设置为1，不相等的像素点设置为0
    cv::compare(result,cv::GC_PR_FGD,result,cv::CMP_EQ);    // 将可能是前景的像素点设置为前景(cv::CMP_EQ:比较的两个值相等)
    result = result & 1;    // 将确定是前景的像素点也设置为前景
    cv::Mat foreground(image.size(),CV_8UC3,cv::Scalar(255,255,255));   // 前景图像
    // src.copyTo(dst, mask)，将image中result为1的像素点复制到foreground中
    image.copyTo(foreground,result);    // 将前景图像复制到前景图像中
    cv::imwrite("../workspace/dog_foreground.png",foreground);
}







