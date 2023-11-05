#include <stdio.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"


class VideoProcessor {
public:
    /*===========================================设置函数(处理相关)===========================================*/
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
        long fnumber = static_cast<long>(capture.get(cv::CAP_PROP_POS_FRAMES)); 
        return fnumber; 
    }

    long getFrameRate(){
        long rate = capture.get(cv::CAP_PROP_FPS);
        return rate;
    }
    
    /*===========================================设置函数(输出相关)===========================================*/
    // 设置输出视频文件
    // 默认情况下会使用与输入视频相同的参数
    bool setOutput(const std::string &filename, int codec=0, 
                   double framerate=0.0, bool isColor=true) { 
        outputFile = filename; 
        extension.clear(); 
        if (framerate == 0.0) 
            // 与输入相同
            framerate = getFrameRate(); 
        // 使用与输入相同的编解码器
        if (codec == 0) { 
            codec = getCodec(); 
        } 
        // 打开输出视频
        return writer.open(outputFile,      // 文件名
                           codec,           // 所用的编解码器
                           framerate,       // 视频的帧速率
                           getFrameSize(),  // 帧的尺寸
                           isColor);        // 彩色视频？
    }

    // 设置输出为一系列图像文件(函数重载)
    // 扩展名必须是.jpg 或.bmp 
    bool setOutput(const std::string &filename,         // 前缀
        const std::string &ext,                         // 图像文件的扩展名
        int numberOfDigits=3,                           // 数字的位数
        int startIndex=0) {                             // 开始序号
            // 数字的位数必须是正数
            if (numberOfDigits<0) 
                return false; 
            // 如果是保存图片
            extension = ext; 
            // 文件编号方案中数字的位数
            digits = numberOfDigits; 
            // 从这个序号开始编号
            currentIndex = startIndex; 
            // 如果是保存视频，需要设置视频相关参数
            outputFile = filename; 
            return true; 
    }

    int getCodec() {
        return static_cast<int>(capture.get(cv::CAP_PROP_FOURCC));
    }
    
    cv::Size getFrameSize(){
        int frameWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        return cv::Size(frameWidth, frameHeight);
    }

    // 写输出的帧
    // 可以是视频文件或图像组
    void writeNextFrame(cv::Mat& frame) { 
        if (extension.length()) {           // 如果指定了扩展名，保存图像到本地
            std::stringstream ss; 
            // 组合成输出文件名
            ss << outputFile << std::setfill('0') << std::setw(digits) << currentIndex++ << extension; 
            cv::imwrite(ss.str(), frame);
        } else { 
            // 写入到视频文件                // 如果未指定扩展名，保存视频到本地
            writer.write(frame); 
        } 
    }

    /*===========================================主函数===========================================*/
    /*===================================抓取（并处理）序列中的帧，写成新的视频(optional)===================================*/
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
             // ** 写入到输出的序列 ** 
            if (outputFile.length() != 0) 
                writeNextFrame(output);
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
    // OpenCV 写视频对象
    cv::VideoWriter writer; 
    // 输出文件名
    std::string outputFile; 
    // 输出图像的当前序号
    int currentIndex; 
    // 输出图像文件名中数字的位数
    int digits; 
    // 输出图像的扩展名
    std::string extension;
};

class FrameProcessor { 
public: 
    // 处理方法
    virtual void process(cv:: Mat &input, cv:: Mat &output)= 0; 
};

class BGFGSegmentor : public FrameProcessor { 
public:
    // 主要处理过程包括将当前帧与背景模型做比较，然后更新该模型
    // 处理方法
    void process(cv:: Mat &frame, cv:: Mat &output) override { 
        // 转换成灰度图像
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); 
        // 采用第一帧初始化背景 
        if (background.empty())
            gray.convertTo(background, CV_32F); 
        // 将背景转换成 8U 类型
        background.convertTo(backImage, CV_8U); 
        // 计算图像与背景之间的差异
        cv::absdiff(backImage, gray, foreground); 
        // 在前景图像上应用阈值
        cv::threshold(foreground, output, threshold, 255, cv::THRESH_BINARY_INV); 
        // 累积背景
        cv::accumulateWeighted(gray, background, 
        // alpha*gray + (1-alpha)*background 
        learningRate, // 学习速率
        output); // 掩码
    };

    void setThreshold(int thresh){
        threshold = thresh;
    };

private:
    cv::Mat gray; // 当前灰度图像
    cv::Mat background; // 累积的背景
    cv::Mat backImage; // 当前背景图像
    cv::Mat foreground; // 前景图像
    // 累计背景时使用的学习速率
    double learningRate; 
    int threshold; // 提取前景的阈值
};


int main() {
    // 提取视频中的前景(没啥用，随便看看就行，有时间了再研究)
    VideoProcessor processor; 
    // 创建背景/前景的分割器
    BGFGSegmentor segmentor; 
    segmentor.setThreshold(25); 
    // 打开视频文件
    processor.setInput("bike.avi"); 
    // 设置帧处理对象
    processor.setFrameProcessor(&segmentor); 
    // 声明显示视频的窗口
    processor.displayOutput("Extracted Foreground"); 
    // 用原始帧速率播放视频
    processor.setDelay(1000./processor.getFrameRate()); 
    // 开始处理
    processor.run();

    return 0;
}