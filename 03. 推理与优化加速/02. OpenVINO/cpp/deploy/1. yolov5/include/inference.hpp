#ifndef INFERENCE_HPP
#define INFERENCE_HPP
#include <string>
#include <vector>
#include <memory>
#include <future>


#include "opencv2/opencv.hpp"

struct Result{
	std::vector<std::vector<int>> boxes;
	std::vector<int> labels;
	std::vector<float> scores;
};

class InferInterface{
public:
    virtual std::shared_future<std::vector<Result>> forward(std::vector<cv::Mat> input_images, 
		                                bool log=false) = 0;
    // virtual std::vector<Result> forward(std::vector<cv::Mat> input_images, 
	// 	                                bool log=false) = 0;
};

std::shared_ptr<InferInterface> create_infer(const std::string &file, 
                                             size_t max_det=100);

#endif //INFERENCE_HPP