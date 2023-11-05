#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "imageComparator.hpp"



int main() {
    cv::Mat image = cv::imread("../workspace/dog.png", 1);
    ImageComparator c; 
    c.setReferenceImage(image);
    c.compare(image);
};