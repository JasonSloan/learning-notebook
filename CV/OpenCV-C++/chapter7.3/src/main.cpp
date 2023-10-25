#include <stdio.h>
#include <cmath>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

int main() {
    cv::Mat image = cv::imread("../workspace/road.png", cv::IMREAD_GRAYSCALE);

    // ==================Part1==================
    // // 基础的霍夫变换检测直线（效果不好，建议用后面的改进版的）
    // // 应用 Canny 算法
    // cv::Mat contours;
    // cv::Canny(image, contours, 125, 350);
    // // 用霍夫变换检测直线
    // std::vector<cv::Vec2f> lines;
    // cv::HoughLines(contours, lines, 1,  // 半径为1，代表每一个像素都会检测
    //                M_PI / 180, // 步长，代表每一种角度都会去寻找
    //                60);        // 最小投票数，值越小，检测到的直线就越多
    // std::vector<cv::Vec2f>::const_iterator it = lines.begin();  //
    // lines中存储这半径rho和角度theta(使用极坐标系表示的直线) while (it !=
    // lines.end())
    // {
    //     float rho = (*it)[0];   // 第一个元素是距离 rho
    //     float theta = (*it)[1]; // 第二个元素是角度 theta
    //     if (theta < M_PI / 4. || theta > 3. * M_PI / 4.)    //
    //     theta代表的是与垂线形成的夹角，所以在小于π/4和大于3π/4的是接近90度角
    //     { // 垂直线（大致）
    //         // 直线与第一行的交叉点
    //         cv::Point pt1(rho / cos(theta), 0);
    //         // 直线与最后一行的交叉点
    //         cv::Point pt2((rho - image.rows * sin(theta)) / cos(theta),
    //         image.rows);
    //         // 画白色的线
    //         cv::line(image, pt1, pt2, cv::Scalar(255), 1);
    //     }
    //     else
    //     { // 水平线（大致）
    //         // 直线与第一列的交叉点
    //         cv::Point pt1(0, rho / sin(theta));
    //         // 直线与最后一列的交叉点
    //         cv::Point pt2(image.cols, (rho - image.cols * cos(theta)) /
    //         sin(theta));
    //         // 画白色的线
    //         cv::line(image, pt1, pt2, cv::Scalar(255), 1);
    //     }
    //     ++it;
    // }
    // cv::imwrite("../workspace/hough.png", image);

    // ==================Part2==================
    // 改进版霍夫
    image = cv::imread("../workspace/wheels.png", 0);
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
    // 画直线
    std::vector<cv::Vec4i>::const_iterator it2 = lines.begin();
    while (it2 != lines.end()) {
        cv::Point pt1((*it2)[0], (*it2)[1]);
        cv::Point pt2((*it2)[2], (*it2)[3]);
        cv::line(image, pt1, pt2, cv::Scalar(255, 255, 255));
        ++it2;
    };
    cv::imwrite("../workspace/hough.png", image);

    // ==================Part3==================
    // 霍夫圆(效果不好，需要调一调参数)
    // 检测霍夫圆之前必须先做平滑
    cv::GaussianBlur(image, image, cv::Size(5, 5), 1.5);
    std::vector<cv::Vec3f> circles;
    // HoughCircles方法将canny检测与霍夫变换集成在这个API中了
    cv::HoughCircles(image, circles, cv::HOUGH_GRADIENT,
                     2,    // 累加器分辨率（图像尺寸/2）
                     50,   // 两个圆之间的最小距离
                     200,  // Canny 算子的高阈值
                     500,  // 最少投票数
                     50,
                     100);  // 最小和最大半径
    std::vector<cv::Vec3f>::const_iterator itc = circles.begin();
    while (itc != circles.end()) {
        cv::circle(image, cv::Point((*itc)[0], (*itc)[1]),  // 圆心
                   (*itc)[2],                               // 半径
                   cv::Scalar(255),                         // 颜色
                   2);                                      // 厚度
        ++itc;
    }
    cv::imwrite("../workspace/hough_circle.png", image);
    printf("Done!\n");
}