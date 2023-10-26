#include <stdio.h>
#include <cmath>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

int main() {
    cv::Mat image = cv::imread("../workspace/road.png", cv::IMREAD_GRAYSCALE);
    double minLength = 100;  // 线的最短长度
    double maxGap = 20;      // 线与线之间最小的间隔
    int minVote = 60;  // 最小投票数（该值越小，检测到的直线越少）
    double deltaRho = 1;  // 检测半径，1代表每个像素都检测
    double deltaTheta = M_PI / 180;  // 检测角度，该值代表所有角度都检测
    cv::Mat contours;
    cv::Canny(image, contours, 125, 350);
    std::vector<cv::Vec4i> lines;
    // 改进版霍夫，lines中每个元素存储4个值，分别是直线的两个端点的x、y值
    cv::HoughLinesP(contours, lines, deltaRho, deltaTheta, minVote, minLength,
                    maxGap);
    int n = 0;  // 选用直线 0
    // 黑色图像
    cv::Mat oneline(contours.size(), CV_8U, cv::Scalar(0));
    // 白色直线
    cv::line(oneline, cv::Point(lines[n][0], lines[n][1]),
             cv::Point(lines[n][2], lines[n][3]), cv::Scalar(255),
             3);  // 直线宽度
    // 轮廓与白色直线进行“与”运算
    cv::bitwise_and(contours, oneline, oneline);
    std::vector<cv::Point> points;
    // 迭代遍历像素，得到所有点的位置
    for (int y = 0; y < oneline.rows; y++) {
        // 行 y
        uchar* rowPtr = oneline.ptr<uchar>(y);
        for (int x = 0; x < oneline.cols; x++) {
            // 列 x
            // 如果在轮廓上
            if (rowPtr[x]) {
                points.push_back(cv::Point(x, y));
            }
        }
    }
    cv::Vec4f line;
    cv::fitLine(points, line,
                cv::DIST_L2,  // 距离类型
                0,            // L2 距离不用这个参数
                0.01, 0.01);  // 精度
    int x0 = line[2];         // 直线上的一个点
    int y0 = line[3];
    int x1 = x0 + 100 * line[0];  // 加上长度为 100 的向量
    int y1 = y0 + 100 * line[1];  // （用单位向量生成）
    // 绘制这条线
    cv::line(image, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(255, 255, 255));  // 颜色和宽度
    cv::imwrite("../workspace/fit_line.png", image);
    printf("Done!\n");
}
