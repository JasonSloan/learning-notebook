#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>

int main(){
    cv::Mat image;
    printf("The rows of the image:%d\n",image.rows);
    printf("The cols of the image:%d\n",image.cols);
    if(image.empty()){
        printf("The image is empty!\n");
    }
    // cv::namedWindow("Image");
    // cv::imshow("Image",image);
    image = cv::imread("../workspace/dog.png", cv::IMREAD_COLOR);
    cv::Mat result;
    // cv::Mat result(image.rows, image.cols, image.type());
    cv::flip(image,result,1);
    cv::imwrite("../workspace/dog_flip.png",result);
    printf("The channels of the image is %d\n",image.channels());
    printf("The type of the image pixel is %d\n",image.type());
    printf(image.type() == CV_8UC3 ? "The image is 8UC3\n" : "The image is not 8UC3\n");
    return 0;
}