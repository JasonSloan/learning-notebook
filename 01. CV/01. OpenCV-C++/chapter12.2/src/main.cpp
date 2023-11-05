#include <stdio.h>
#include <cmath>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

int main() {
    // 读取视频，跳转到指定位置（可以通过指定帧数、相对位置、时间位置三种方式跳转）
    // 打开视频文件
    cv::VideoCapture capture("../workspace/scene.mp4");
    // 检查视频是否成功打开
    if (!capture.isOpened())
        return 1;
    // 取得帧速率
    double rate = capture.get(cv::CAP_PROP_FPS);
    printf("帧速率是%lf\n", rate);
    long t = static_cast<long>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    printf("总帧数是%ld\n", t);
    // 跳转到第 100 帧
    double position = 100.0;
    capture.set(cv::CAP_PROP_POS_FRAMES, position);
    printf("已将视频跳转到第%d帧\n", static_cast<int>(position));
    /*还可以用 cv::CAP_PROP_POS_MSEC 以毫秒为单位指定位置，或者用 cv::CAP_PROP_POS_AVI_RATIO 
      指定视频内部的相对位置（0.0 表示视频开始位置，1.0 表示结束位置）。如果参数设
      置成功，函数会返回 true。*/
    bool stop(false);
    // 当前视频帧
    cv::Mat frame;
    // 根据帧速率计算帧之间的等待时间，单位为 ms
    int delay = 1000 / rate;
    // 循环遍历视频中的全部帧
    while (!stop) {
        // 读取下一帧（如果有）
        if (!capture.read(frame))
            break;
    }
    // 关闭视频文件
    // 不是必须的，因为类的析构函数会调用
    capture.release();
    return 0;
}