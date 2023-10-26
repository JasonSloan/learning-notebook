#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>

class WatershedSegmenter {
/*
    用分水岭算法分割图像的
    原理是从高度 0 开始逐步用洪水淹没图像。当“水”的高度逐步增加时（到 1、2、3 等），会形
    成聚水的盆地。随着盆地面积逐步变大，两个盆地的水最终会汇合到一起。这时就要创建一个分水
    岭，用来分割这两个盆地。当水位达到最大高度时，创建的盆地和分水岭就组成了分水岭分割图。
    可以想象，在水淹过程的开始阶段会创建很多细小的独立盆地。当所有盆地汇合时，就会创
    建很多分水岭线条，导致图像被过度分割。要解决这个问题，就要对这个算法进行修改，使水淹
    过程从一组预先定义好的标记像素开始。每个用标记创建的盆地，都按照初始标记的值加上标签。
    如果两个标签相同的盆地汇合，就不创建分水岭，以避免过度分割。调用 cv::watershed 函数
    时就执行了这些过程。输入的标记图像会被修改，用以生成最终的分水岭分割图。输入的标记图
    像可以含有任意数值的标签，未知标签的像素值为 0。标记图像的类型选用 32 位有符号整数，
    以便定义超过 255 个的标签。另外，可以把分水岭的对应像素设为特殊值-1。
*/ 

public: 
    void setMarkers(const cv::Mat& markerImage) { 
        // 转换成整数型图像
        markerImage.convertTo(markers, CV_32S); 
    } 

    cv::Mat process(const cv::Mat &image) { 
        // 应用分水岭
        cv::watershed(image, markers); 
        return markers; 
    }
private: 
    cv::Mat markers; 
};

int main(){
    // ？没work。还未找到原因
    cv::Mat image = cv::imread("../workspace/dog.png", cv::IMREAD_GRAYSCALE);
    cv::Mat binary;
    cv::threshold(image, binary, 114, 255, cv::THRESH_BINARY_INV);
    // 消除噪声和细小物体
    cv::Mat fg; 
    cv::erode(binary, fg, cv::Mat(), cv::Point(-1,-1), 4); 
    // 创建分水岭分割类的对象
    WatershedSegmenter segmenter; 
    // 标识不含物体的图像像素
    cv::Mat bg; 
    cv::dilate(binary, bg, cv::Mat(), cv::Point(-1,-1), 4); 
    cv::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV); 
    // 创建标记图像
    cv::Mat markers(binary.size(), CV_32S, cv::Scalar(0)); 
    markers = fg + bg; 
    // 设置标记图像，然后执行分割过程
    segmenter.setMarkers(markers); 
    segmenter.process(image); 
}