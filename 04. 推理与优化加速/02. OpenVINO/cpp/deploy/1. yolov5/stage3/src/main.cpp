#include <stdio.h>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <future>

#include "spdlog/logger.h"                              // spdlog日志相关
#include "spdlog/spdlog.h"                              // spdlog日志相关
#include "opencv2/opencv.hpp"
#include "tqdm.hpp"
#include "utils.hpp"
#include "inference.h"

using namespace std;
using namespace cv;


int main(){
	print_avaliable_devices();
	// !注意:max-det是单张图片最多允许检测的目标数, 需要和训练部分add-postprocess.py中设置的保持一致, 可以不指定, 已写为默认值100
	int n_iter = 100;
	int batch_size = 2;
	int max_det = 100;
	string device = "GPU";
	string modelPath = "weights/multi-batch/ov-fp16-nonms.xml";
	auto inferController = create_infer(modelPath, max_det, device, false);

	// set inputs path
	string path = "inputs/images";
	// string path = "inputs/videos/ch02_20230715000000_part0.mp4";

	// prepare data
	int imgcount;
	vector<vector<Mat>> m_imgs;
	vector<vector<string>> save_paths;
	vector<vector<string>> unique_ids;
	collect_data(path, batch_size, imgcount, m_imgs, save_paths, unique_ids);
	int n_batch = imgcount / batch_size;
	Input i_imgs[3][2];									// ! 必须显示指定i_imgs这两个维度(需要手动计算一下)
	memset(&i_imgs, 0, sizeof(i_imgs));
	transfer_data(m_imgs, i_imgs, unique_ids);			// transfer data from cv::Mat to Input struct

	// infer
	spdlog::info("----> Start infering......");
	for (int n = 0; n < n_iter; ++n) {
		for (int i : tq::trange(n_batch)){													// i is batch idx, imgs[i] is number i batch 
			shared_future<vector<Result>> fut = inferController->forward(i_imgs[i], batch_size, true);		// true for log or not
			vector<Result> results = fut.get();													// asynchronous get result
			// draw_rectangles(results, m_imgs[i], save_paths[i]);									// draw rectangles and save image
		}
	}
	printf("\n");
	spdlog::info("----> Infer successfully!");

	// print elapsed time
	auto records = inferController->get_records();
	auto avg_preprocess_time = mean(records[0]) / float(batch_size);
	auto avg_infer_time = mean(records[1]) / float(batch_size);
	auto avg_postprocess_time = mean(records[2]) / float(batch_size);
	auto avg_total_time = avg_preprocess_time + avg_infer_time + avg_postprocess_time;
	spdlog::info("----> Average time cost: preprocess {} ms, infer {} ms, postprocess {} ms, total {} ms", 
		avg_preprocess_time, avg_infer_time, avg_postprocess_time, avg_total_time);
    return 0;
};