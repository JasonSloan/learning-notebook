#include <stdio.h>
#include <string>
#include <vector>
#include <chrono>

#include "opencv2/opencv.hpp"
#include "tqdm.hpp"
#include "utils.hpp"
#include "inference.hpp"


using namespace std;
using namespace cv;

int main(){
	// !注意:max-det是单张图片最多允许检测的目标数, 需要和训练部分add-postprocess.py中设置的保持一致, 可以不指定, 已写为默认值100
	size_t batch_size = 10;
	size_t max_det = 100;
	string modelPath = "weights/multi-batch/ov.xml";
	auto inferController = create_infer(modelPath);

	// set inputs path
	string path = "inputs/images";
	// string path = "inputs/videos/ch02_20230715000000_part0.mp4";
	bool is_video = ends_with(path, ".mp4");

	// prepare and infer
	vector<vector<Mat>> imgs;
	vector<vector<string>> save_paths;
	if (!is_video){
		vector<string> files;
		bool success = listdir(path, files);
		// push back imgs into a vector
		int batch_idx = 0;
		int num_batch = ceil(float(files.size()) / float(batch_size));
		imgs.resize(num_batch);
		save_paths.resize(num_batch);
		cout << "Start reading and pushing files..." << endl;
		for (int i : tq::trange(files.size())){
			cv::Mat img = cv::imread(files[i]);
			if (img.empty()) {
				printf("Unable to read image %s\n", files[i].c_str());
				continue;
			}
			if ((i != 0) && (i % batch_size == 0))		// if read successfully then compute the batch_idx
				batch_idx++;
			imgs[batch_idx].push_back(img);
			std::string filename = getFileName(files[i]);
			string save_path = "outputs/" + filename;
			save_paths[batch_idx].push_back(save_path);
		}
	}

	else {
		cv::VideoCapture cap(path);
		if (!cap.isOpened()) {
			printf("Unable to open video %s", path.c_str());
			return -1;
		}
		cv::Mat frame;
		int total = 0;
		int broken = 0;
		int batch_idx = 0;
		int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
		int num_batch = ceil(float(frame_count) / float(batch_size));
		imgs.resize(num_batch);
		save_paths.resize(num_batch);
		cout << "Start reading and pushing files..." << endl;
		for (int i : tq::trange(frame_count)) {
			cap >> frame;
			if (frame.empty()) {
				printf("Unable to read frame %d of video %s\n", total, path.c_str());
				broken++;
				continue;
			}
			if ((i != 0) && (i % batch_size == 0))		// if read successfully then compute the batch_idx
				batch_idx++;
			imgs[batch_idx].push_back(frame);
			string save_path = "outputs/frame_" + std::to_string(total) + ".jpg";
			save_paths[batch_idx].push_back(save_path);
			total++;
		}
		printf("\nRead video %s successfully, total frames: %d, broken frames: %d\n", path.c_str(), total, broken);
		cap.release();
	}

	// infer
	cout << "\nStart infering......" << endl;
	auto start = std::chrono::high_resolution_clock::now();
	for (int i : tq::trange(imgs.size())){										// i is batch idx, imgs[i] is number i batch 
		vector<Result> results = inferController->forward(imgs[i], true);		// true for log or not	
		// draw_rectangles(results, imgs[i]);
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto durations = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("\nAverage time consuming per image: %.2f ms\n", durations.count() / 1000. / float(imgs.size() * batch_size));

    return 0;
};