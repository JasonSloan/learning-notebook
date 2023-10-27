#include <stdio.h>
#include <cmath>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

class VideoProcessor {
public:
    /*===========================================设置函数===========================================*/
    // 设置针对每一帧调用的回调函数（处理每一帧的函数）
    // 将函数指针传递给函数的用法：frameProcessingCallback代表回调函数的指针，它指向的函数接受两个cv::Mat的参数，返回值为void
    void setFrameProcessor(void (*frameProcessingCallback)(cv::Mat&, cv::Mat&)) {
        process = frameProcessingCallback;
        callIt = true;
    }

    // 打开视频文件
    bool setInput(std::string filename) {
        fnumber = 0;
        // 防止已经有资源与 VideoCapture 实例关联
        capture.release();
        // 打开视频文件
        return capture.open(filename);
    }

    // 设置用于显示输入的帧的窗口名字
    void displayInput(std::string wn) {
        windowNameInput = wn;
        cv::namedWindow(windowNameInput);
    }

    // 设置用于显示处理过的帧的窗口名字
    void displayOutput(std::string wn) {
        windowNameOutput = wn;
        cv::namedWindow(windowNameOutput);
    }

    // 结束处理
    void stopIt() { stop = true; }

    // 处理过程是否已经停止？
    bool isStopped() { return stop; }

    // 捕获设备是否已经打开？
    bool isOpened() { return capture.isOpened(); }

    // 设置帧之间的延时，
    // 0 表示每一帧都等待，
    // 负数表示不延时
    void setDelay(int d) { delay = d; }

    // 读取下一帧
    bool readNextFrame(cv::Mat& frame) {
        return capture.read(frame);
    }

    // 设置是否需要在处理完一定数量的帧后就结束
    void stopAtFrameNo(long frame) { 
        frameToStop= frame; 
    } 

    // 返回下一帧的编号
    long getFrameNumber() { 
        // 从捕获设备获取信息
        long fnumber= static_cast<long>(capture.get(cv::CAP_PROP_POS_FRAMES)); 
        return fnumber; 
    }

    long getFrameRate(){
        long rate = capture.get(cv::CAP_PROP_FRAME_COUNT);
        return rate;
    }

    /*===========================================主函数===========================================*/
    // 抓取（并处理）序列中的帧
    void run() {
        // 当前帧
        cv::Mat frame;
        // 输出帧
        cv::Mat output;
        // 如果没有设置捕获设备
        if (!isOpened())
            return;
        stop = false;
        while (!isStopped()) {
            // 读取下一帧（如果有）
            if (!readNextFrame(frame))
                break;
            // 显示输入的帧
            if (windowNameInput.length() != 0)
                cv::imshow(windowNameInput, frame);
            // 调用处理函数
            if (callIt) {
                // 处理帧
                process(frame, output);
                // 递增帧数
                fnumber++;
                // // 保存图片
                // std::stringstream filenameStream;
                // filenameStream << "../workspace/" << fnumber << ".png";
                // std::string filename = filenameStream.str();
                // cv::imwrite(filename, output);
            } else {
                // 没有处理
                output = frame;
            }
            // 显示输出的帧
            if (windowNameOutput.length() != 0)
                cv::imshow(windowNameOutput, output);
            // 产生延时
            if (delay >= 0 && cv::waitKey(delay) >= 0)
                stopIt();
            // 检查是否需要结束
            if (frameToStop >= 0 && getFrameNumber() == frameToStop)
                stopIt();
        };
    }

private:
    // OpenCV 视频捕获对象
    cv::VideoCapture capture; 
    // 处理每一帧时都会调用的回调函数
    void (*process)(cv::Mat&, cv::Mat&);
    // 布尔型变量，表示该回调函数是否会被调用
    bool callIt{false};
    // 输入窗口的显示名称
    std::string windowNameInput;
    // 输出窗口的显示名称
    std::string windowNameOutput;
    // 帧之间的延时
    int delay;
    // 已经处理的帧数
    long fnumber;
    // 达到这个帧数时结束
    long frameToStop;
    // 结束处理
    bool stop;
};

// 回调函数
void canny(cv::Mat& img, cv::Mat& out) { 
    // 转换成灰度图像
    if (img.channels()==3) 
        cv::cvtColor(img, out, cv::COLOR_BGR2GRAY); 
    // 计算 Canny 边缘
    cv::Canny(out, out, 100, 200); 
    // 反转图像
    cv::threshold(out, out, 128, 255, cv::THRESH_BINARY_INV); 
};

int main() {
    // 创建实例
    VideoProcessor processor; 
    // 打开视频文件
    processor.setInput("../workspace/scene.mp4"); 
    // // 声明显示视频的窗口
    // processor.displayInput("Current Frame"); 
    // processor.displayOutput("Output Frame"); 
    // 用原始帧速率播放视频
    processor.setDelay(1000./processor.getFrameRate()); 
    // 设置处理帧的回调函数
    processor.setFrameProcessor(canny); 
    // 开始处理
    processor.run();
    return 0;
}