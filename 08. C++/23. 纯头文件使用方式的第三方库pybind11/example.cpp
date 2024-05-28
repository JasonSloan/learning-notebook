#include <cstring>
#include "opencv2/opencv.hpp"
#include "pybind11.hpp"

using namespace std;
using namespace cv;
namespace py = pybind11;

int main(){
	// todo: 未运行成功
    Mat img = imread("dog.jpg");
    auto numel = img.rows * img.cols * img.channels();
    py::array imageArray;
    memcpy((uchar*)(imageArray.data(0)), img.data, numel);
};