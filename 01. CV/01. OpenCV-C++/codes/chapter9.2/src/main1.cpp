#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main() {
    /*模板匹配
    只能匹配到和模板大小一样的区域*/

    // 读取输入图像
    cv::Mat image = cv::imread("/root/study/opencv/workspace/1.jpg", 1);
    cv::Mat target = cv::imread("/root/study/opencv/workspace/crop1.jpg", 1);
    cv::Mat result;
    // 定义搜索区域(整图搜索)
    cv::Mat roi(image);
    // 进行模板匹配
    cv::matchTemplate(roi,             // 搜索区域
                      target,          // 模板
                      result,          // 结果
                      cv::TM_SQDIFF);  // 相似度
    // 找到最相似的位置
    double minVal, maxVal;
    cv::Point minPt, maxPt;
    cv::minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt);
    // 在相似度最高的位置绘制矩形
    // 本例中为 minPt
    cv::rectangle(roi, cv::Rect(minPt.x, minPt.y, target.cols, target.rows), 255);
    // 保存结果
    cv::imwrite("/root/study/opencv/workspace/template.jpg", image);
    return 0;
}
