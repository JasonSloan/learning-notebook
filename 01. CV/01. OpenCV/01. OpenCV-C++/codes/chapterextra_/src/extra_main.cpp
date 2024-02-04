#include <stdio.h>
#include <memory>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

int main() {
    // 将OpenCv读取的BGRBGRBGRBGRBGRBGR...格式的数据转成RRRRRRGGGGGGGBBBBBB格式的浮点数据
    cv::Mat image = cv::imread("../workspace/3.jpg", cv::IMREAD_COLOR);
    int channels = image.channels();
    int cols = image.cols;
    int rows = image.rows;
    int image_area = cols * rows;  // 转完后每个通道应该有image_area个元素
    int numel = channels * cols * rows * sizeof(float);  // 转完后总共占用字节数
    float* data = new float[numel];                      // 分配内存
    float* phost_b = data + image_area * 0;              // b通道的地址
    float* phost_g =
        data + image_area * 1;  // g通道的地址，等于b通道偏移image_area个位置
    float* phost_r =
        data + image_area * 2;  // r通道的地址，等于g通道偏移image_area个位置
    unsigned char* pimage = image.data;  // 原图图像首地址
    for (int i = 0; i < image_area; ++i, pimage += 3)  // 每次循环原图移动3个位置（每次移动BGR个位置），r、g、b移动一个位置
    {
        // 这里实现了bgr->rgb的变换
        *phost_r++ = pimage[0] / 255.0f;  // 每次循环取出的pimage[0]是b通道的值
        *phost_g++ = pimage[1] / 255.0f;  // 每次循环取出的pimage[1]是g通道的值
        *phost_b++ = pimage[2] / 255.0f;  // 每次循环取出的pimage[2]是r通道的值
    }
    delete[] data;
    printf("Done!\n");
    return 0;
}
