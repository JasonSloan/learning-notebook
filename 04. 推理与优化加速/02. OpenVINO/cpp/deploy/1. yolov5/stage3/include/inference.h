#ifndef INFERENCE_HPP
#define INFERENCE_HPP


#include <string>
#include <vector>
#include <memory>
#include <future>


struct Result{
	std::string unique_id;
	std::vector<std::vector<int>> boxes;
	std::vector<int> labels;
	std::vector<float> scores;
};

struct Input{
	std::string unique_id;
	unsigned char* data;
	int height;
	int width;
};

class InferInterface{
public:
    virtual std::shared_future<std::vector<Result>> forward(Input* inputs, int& n_images, bool inferLog=false) = 0;
	virtual std::vector<std::vector<float>> get_records() = 0;		// 计时相关, 可删
};

std::shared_ptr<InferInterface> create_infer(std::string &file, int max_det, std::string& device, bool modelLog);

#endif