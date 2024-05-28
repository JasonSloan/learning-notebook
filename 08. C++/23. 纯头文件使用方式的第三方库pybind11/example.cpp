#include "opencv2/opencv.hpp"
#include "pybind11.hpp"

using namespace std;
namespace py = pybind11;

int main(){
    py::array& imageArray;
    cv::Mat cvimage(imageArray.shape(0), imageArray.shape(1), CV_8UC3, (unsigned char*)imageArray.data(0));
};