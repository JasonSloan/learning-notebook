#include "opencv2/opencv.hpp"

class RobustMatcher {
   private:
    // 特征点检测器对象的指针
    cv::Ptr<cv::FeatureDetector> detector;
    // 特征描述子提取器对象的指针
    cv::Ptr<cv::DescriptorExtractor> descriptor;
    int normType;
    float ratio;        // 第一个和第二个 NN 之间的最大比率
    bool refineF;       // 如果等于 true，则会优化基础矩阵
    bool refineM;       // 如果等于 true，则会优化匹配结果
    double distance;    // 到极点的最小距离
    double confidence;  // 可信度（概率）

   public:
    RobustMatcher(const cv::Ptr<cv::FeatureDetector>& detector,
                  const cv::Ptr<cv::DescriptorExtractor>& descriptor =
                      cv::Ptr<cv::DescriptorExtractor>())
        : detector(detector),
          descriptor(descriptor),
          normType(cv::NORM_L2),
          ratio(0.8f),
          refineF(true),
          refineM(true),
          confidence(0.98),
          distance(1.0) {
        // 这里使用关联描述子
        if (!this->descriptor) {
            this->descriptor = this->detector;
        }
    }

    // 用 RANSAC 算法匹配特征点
    // 返回基础矩阵和输出的匹配项
    cv::Mat match(cv::Mat& image1,
                  cv::Mat& image2,                        // 输入图像
                  std::vector<cv::DMatch>& matches,       // 输出匹配项
                  std::vector<cv::KeyPoint>& keypoints1,  // 输出关键点
                  std::vector<cv::KeyPoint>& keypoints2) {
        // 1.检测特征点
        detector->detect(image1, keypoints1);
        detector->detect(image2, keypoints2);
        // 2.提取特征描述子
        cv::Mat descriptors1, descriptors2;
        descriptor->compute(image1, keypoints1, descriptors1);
        descriptor->compute(image2, keypoints2, descriptors2);
        // 3.匹配两幅图像描述子
        // （用于部分检测方法）
        // 构造匹配类的实例（带交叉检查）
        cv::BFMatcher matcher(normType,  // 差距衡量
                              true);     // 交叉检查标志
        // 匹配描述子
        std::vector<cv::DMatch> outputMatches;
        matcher.match(descriptors1, descriptors2, outputMatches);
        // 4.用 RANSAC 算法验证匹配项
        cv::Mat fundamental =
            ransacTest(outputMatches, keypoints1, keypoints2, matches);
        // 返回基础矩阵
        return fundamental;
    }

    // 用 RANSAC 算法获取优质匹配项
    // 返回基础矩阵和匹配项
    cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
                       std::vector<cv::KeyPoint>& keypoints1,
                       std::vector<cv::KeyPoint>& keypoints2,
                       std::vector<cv::DMatch>& outMatches) {
        // 将关键点转换为 Point2f 类型
        std::vector<cv::Point2f> points1, points2;
        for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
             it != matches.end(); ++it) {
            // 获取左侧关键点的位置
            points1.push_back(keypoints1[it->queryIdx].pt);
            // 获取右侧关键点的位置
            points2.push_back(keypoints2[it->trainIdx].pt);
        }
        // 用 RANSAC 计算 F 矩阵
        std::vector<uchar> inliers(points1.size(), 0);
        cv::Mat fundamental =
            cv::findFundamentalMat(points1,
                                   points2,  // 匹配像素点
                                   inliers,  // 匹配状态（inlier 或 outlier)
                                   cv::FM_RANSAC,  // RANSAC 算法
                                   distance,       // 到对极线的距离
                                   confidence);    // 置信度
        return fundamental;
    }
};

int main() {
    /*
        对两幅图像进行特征点匹配, 得到很多对匹配的特征点;
        对匹配的特征点使用ransac算法进行优质特征点的选择,
        从而得到一个比较好的基础矩阵
    */
    auto image1 = cv::imread("workspace/1.jpg");
    auto image2 = cv::imread("workspace/2.jpg");
    // 准备匹配器（用默认参数）
    // SIFT 检测器和描述子
    RobustMatcher rmatcher(cv::SIFT::create(250));
    // 匹配两幅图像
    std::vector<cv::DMatch> matches;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat fundamental =
        rmatcher.match(image1, image2, matches, keypoints1, keypoints2);
    return 0;
}