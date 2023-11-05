#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>



int main(){
    // 图像融合
    // c[i]= a[i]+b[i]; 
    cv::add(imageA,imageB,resultC); 
    // c[i]= a[i]+k; 
    cv::add(imageA,cv::Scalar(k),resultC); 
    // c[i]= k1*a[i]+k2*b[i]+k3; 
    cv::addWeighted(imageA,k1,imageB,k2,k3,resultC); 
    // c[i]= k*a[i]+b[i]; 
    cv::scaleAdd(imageA,k,imageB,resultC); 
    /*
    有些函数还可以指定一个掩码：
    如果(mask[i]) c[i]= a[i]+b[i]; 
    cv::add(imageA,imageB,resultC,mask); 
    使用掩码后，操作就只会在掩码值非空的像素上执行（掩码必须是单通道的）。看一下
    cv::subtract、cv::absdiff、cv::multiply 和 cv::divide 等函数的多种格式。此外还
    有位运算符（对像素的二进制数值进行按位运算）cv::bitwise_and、cv::bitwise_or、
    cv::bitwise_xor 和 cv::bitwise_not。cv::min 和 cv::max 运算符也非常实用，它们能
    找到每个元素中最大或最小的像素值。
    在所有场合都要使用 cv::saturate_cast 函数（详情请参见 2.6 节），以确保结果在预定
    的像素值范围之内（避免上溢或下溢）。
    这些图像必定有相同的大小和类型（如果与输入图像的大小不匹配，输出图像会重新分配）。
    由于运算是逐个元素进行的，因此可以把其中的一个输入图像用作输出图像。
    还有运算符使用单个输入图像，它们是 cv::sqrt、cv::pow、cv::abs、cv::cuberoot、
    cv::exp 和 cv::log。事实上，无论需要对图像像素做什么运算，OpenCV 几乎都有相应的函数。
    */

   /*
    // 通道分割合并
    // 创建三幅图像的向量
    std::vector<cv::Mat> planes; 
    // 将一个三通道图像分割为三个单通道图像
    cv::split(image1,planes); 
    // 加到蓝色通道上
    planes[0]+= image2; 
    // 将三个单通道图像合并为一个三通道图像
    cv::merge(planes,result); 
   */
}