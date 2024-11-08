#include <stdio.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <dirent.h>     // opendir和readdir包含在这里
#include <cstring>      // strcmp包含在这里
#include <memory>

#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include "tqdm.hpp"

using namespace std;
using namespace cv;


const vector<Scalar> COLORS = {
	{255, 0, 0},
	{0, 255, 0},
	{0, 0, 255},
	{0, 255, 255}
};

const string LABELS[] = {"head", "helmet", "person", "lookout"};

struct Result{
	std::vector<vector<int>> boxes;
	std::vector<int> labels;
	std::vector<float> scores;
};

void print_avaliable_devices() {
	ov::Core core;
	vector<string> availableDevices = core.get_available_devices();
	for (int i = 0; i < availableDevices.size(); i++) {
		printf("supported device name : %s \n", availableDevices[i].c_str());
	}
}

void build_model(const string& modelPath, ov::InferRequest & request){
	ov::Core core;
	ov::CompiledModel model = core.compile_model(modelPath, "AUTO");
	request = model.create_infer_request();
}

void preprocess(vector<Mat>& batched_imgs, 
				ov::InferRequest& request, 
				vector<int>& batched_pad_w, 
				vector<int>& batched_pad_h, 
				vector<float>& batched_scale_factors, 
				size_t& batch_size, 
				size_t& max_det, 
				bool log=true){
	// set input & ouput shape for dynamic batch 
	ov::Tensor input_tensor = request.get_input_tensor();
	ov::Shape input_shape = input_tensor.get_shape();
    input_shape[0] = batch_size; // Set the batch size in the input shape
	input_tensor.set_shape(input_shape);
	size_t input_channel = input_shape[1];
	size_t input_height = input_shape[2];
	size_t input_width = input_shape[3];
		
	ov::Tensor output_tensor = request.get_output_tensor();
	ov::Shape output_shape = output_tensor.get_shape();
	output_shape[0] = batch_size * max_det;
	output_tensor.set_shape(output_shape);

	if (log) {
		printf("Model input shape: %ld x %ld x %ld x %ld\n", batch_size, input_channel, input_height, input_width);
		printf("Model max output shape: %ld x %ld\n", output_shape[0], output_shape[1]);
	}

	// reize and pad
	for (int i = 0; i < batched_imgs.size(); ++i){
		Mat& img = batched_imgs[i];
		int img_height = img.rows;
		int img_width = img.cols;
		int img_channels = img.channels();

		float scale_factor = min(static_cast<float>(input_width) / static_cast<float>(img.cols),
						static_cast<float>(input_height) / static_cast<float>(img.rows));
		int img_new_w_unpad = img.cols * scale_factor;
		int img_new_h_unpad = img.rows * scale_factor;
		int pad_w = (input_width - img_new_w_unpad) / 2;		// todo: lack of one pixel may occur: eg.1079-->640
		int pad_h = (input_height - img_new_h_unpad) / 2;
		cv::resize(img, img, cv::Size(img_new_w_unpad, img_new_h_unpad));
		cv::copyMakeBorder(img, img, pad_h, pad_h, pad_w, pad_w, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
		batched_scale_factors.push_back(scale_factor);
		batched_pad_w.push_back(pad_w);
		batched_pad_h.push_back(pad_h);
	}

	// BGR-->RGB & HWC-->CHW & /255. & transfer data to input_tensor
	float* input_data_host = input_tensor.data<float>();
	float* i_input_data_host;
	int img_area = input_height * input_width;
	for (int i = 0; i < batched_imgs.size(); ++i){
		i_input_data_host = input_data_host + img_area * 3 * i;
		unsigned char* pimage = batched_imgs[i].data;
		float* phost_b = i_input_data_host + img_area * 0;
		float* phost_g = i_input_data_host + img_area * 1;
		float* phost_r = i_input_data_host + img_area * 2;
		for(int j = 0; j < img_area; ++j, pimage += 3){
			*phost_r++ = pimage[0] / 255.0f ;
			*phost_g++ = pimage[1] / 255.0f;
			*phost_b++ = pimage[2] / 255.0f;
		}
	}
}

void do_infer(ov::InferRequest & request){
	request.infer();
}

void postprocess(vector<Result>& results, 
			     ov::InferRequest & request, 				
				 vector<int>& batched_pad_w, 
				 vector<int>& batched_pad_h, 
				 vector<float>& batched_scale_factors, 
				 size_t& batch_size, 
				 bool log=true) {
	ov::Tensor output = request.get_output_tensor();
	size_t num_boxes = output.get_shape()[0];
	// xyxy + conf + cls_id + image_idx
	size_t num_dim = output.get_shape()[1];

	if (log) {
		printf("Current batch output shape: %ld x %ld \n", num_boxes, num_dim);
	}

	cv::Mat prob(num_boxes, num_dim, CV_32F, (float*)output.data());
	for (int i = 0; i < num_boxes; i++) {
		float conf = prob.at<float>(i, 4);
		int image_idx = static_cast<int>(prob.at<float>(i, 6));
		int class_id = static_cast<int>(prob.at<float>(i, 5));

		int pad_w = batched_pad_w[image_idx];
		int pad_h = batched_pad_h[image_idx];
		float scale_factor = batched_scale_factors[image_idx];
		int predx1 = std::round((prob.at<float>(i, 0) - float(pad_w)) / scale_factor);
		int predy1 = std::round((prob.at<float>(i, 1) - float(pad_h)) / scale_factor);
		int predx2 = std::round((prob.at<float>(i, 2) - float(pad_w)) / scale_factor);
		int predy2 = std::round((prob.at<float>(i, 3) - float(pad_h)) / scale_factor);
		
		vector<int> cords = {predx1, predy1, predx2, predy2};
		results[image_idx].boxes.emplace_back(cords);
		results[image_idx].labels.emplace_back(class_id);
		results[image_idx].scores.emplace_back(conf);
		if (log) {
			printf("image_idx: %d, class_id: %d, conf: %.2f, xyxy: %d %d %d %d\n", image_idx, class_id, conf, predx1, predy1, predx2, predy2);
		}
	}
}

void draw_rectangles(vector<Result>& results, vector<Mat>& im0s){
	for (int i = 0; i < results.size(); ++i) {
		Result result = results[i];
		Mat& im0 = im0s[i];
		for (int j = 0; j < result.boxes.size(); j++) {
			cv::rectangle(
				im0, 
				cv::Point(result.boxes[j][0], result.boxes[j][1]), 
				cv::Point(result.boxes[j][2], result.boxes[j][3]), 
				COLORS[result.labels[j]], 
				5, 8, 0
				);
			// cv::putText(im0, LABELS[result.labels[i]], cv::Point(result.boxes[i][0], result.boxes[i][1]), cv::FONT_HERSHEY_SIMPLEX, 1.4, COLORS[result.labels[i]], 2);
		}
	}

}

void print_consuming_time(std::vector<std::chrono::microseconds>& durations){
	printf("Consuming time: %.2f ms preprocess, %.2f ms infer, %.2f ms postprocess\n", 
			durations[0].count() / 1000.0, durations[1].count() / 1000.0, durations[2].count() / 1000.0);
	durations.clear();
}

void infer(vector<Mat>& batched_imgs, 
		   ov::InferRequest& request, 
		   vector<string>& batched_save_path, 
		   size_t& batch_size, 
		   size_t& max_det, 
		   bool plot=true, 
		   bool log=true) {
	// for visualization
	// todo: only for debug, delete later
	vector<Mat> im0s;
	for (int i = 0; i < batched_imgs.size(); i++){
		im0s.push_back(batched_imgs[i].clone());
	}
			

	// initialize time
	std::vector<std::chrono::microseconds> durations;
	auto start = std::chrono::high_resolution_clock::now();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	// preprocess 
	start = std::chrono::high_resolution_clock::now();
	vector<int> batched_pad_w, batched_pad_h;
	vector<float> batched_scale_factors;
	preprocess(batched_imgs, request, batched_pad_w, batched_pad_h, batched_scale_factors, batch_size, max_det, log);
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	durations.push_back(duration);

	// infer
	start = std::chrono::high_resolution_clock::now();
	do_infer(request);
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	durations.push_back(duration);

	// postprocess
	start = std::chrono::high_resolution_clock::now();
	vector<Result> results;
	results.resize(batch_size);
	postprocess(results, request, batched_pad_w, batched_pad_h, batched_scale_factors, batch_size, log);
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	durations.push_back(duration);

	// draw rectangles
	if (plot == true){
		draw_rectangles(results, im0s);
		for (int i = 0; i < im0s.size(); ++i){
			cv::imwrite(batched_save_path[i], im0s[i]);
		}
	}

	// print time consuming
	if (log == true){
		print_consuming_time(durations);
	}
}

int listdir(string& input,  vector<string>& files_vector) {
    // 打开文件夹
    DIR* pDir = opendir(input.c_str());
    if (!pDir) {
        cerr << "Error opening directory: " << strerror(errno) << endl;
        return 1;
    }
    struct dirent* ptr;
    // 读取文件夹中的文件
    while ((ptr = readdir(pDir)) != nullptr) {
        // strcmp比较两个字符串, 如果不是"."或者".."就继续
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files_vector.push_back(input + "/" + ptr->d_name);
        }
    }

    // 关闭文件夹
    closedir(pDir);
    std::sort(files_vector.begin(), files_vector.end());

	return 0;
}

bool ends_with(const std::string &str, const std::string &ending) {
    if (str.length() >= ending.length()) {
        return str.compare(str.length() - ending.length(), ending.length(), ending) == 0;
    }
    return false;
}

std::string getFileName(const std::string& file_path, bool with_ext=true){
	int index = file_path.find_last_of('/');
	if (index < 0)
		index = file_path.find_last_of('\\');
    std::string tmp = file_path.substr(index + 1);
    if (with_ext)
        return tmp;

    std::string img_name = tmp.substr(0, tmp.find_last_of('.'));
    return img_name;
}

int main(){
	// create model
	string modelPath = "weights/multi-batch/ov.xml";
	ov::InferRequest request;
	build_model(modelPath, request);
	printf("Model load successfully!\n");

	// set batch size and max_det
	// !注意:max-det是单张图片最多允许检测的目标数, 需要和训练部分add-postprocess.py中设置的保持一致
	size_t batch_size = 10;
	size_t max_det = 100;

	// set inputs path
	// string path = "inputs/images";
	string path = "inputs/videos/ch02_20230715000000_part0.mp4";
	bool is_video = ends_with(path, ".mp4");

	// prepare and infer
	if (!is_video){
		vector<string> files;
		bool success = listdir(path, files);
		// push back imgs into a vector
		int batch_idx = 0;
		int num_batch = ceil(float(files.size()) / float(batch_size));
		vector<vector<Mat>> imgs(num_batch);
		vector<vector<string>> save_paths(num_batch);
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

		// infer
		cout << "\nStart infering......" << endl;
		auto start = std::chrono::high_resolution_clock::now();
		for (int i : tq::trange(imgs.size())){		// i is batch idx, imgs[i] is number i batch 
			size_t i_batch_size = imgs[i].size();	// actual batch size for current batch, vary at last batch
			infer(imgs[i], request, save_paths[i], i_batch_size, max_det, true, false);	// plot & log
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto durations = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		printf("\nAverage time consuming per image: %.2f ms\n", durations.count() / 1000. / float(imgs.size() * batch_size));

	} else {
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
		vector<vector<Mat>> imgs(num_batch);
		vector<vector<string>> save_paths(num_batch);
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

		// infer
		cout << "\nStart infering......" << endl;
		auto start = std::chrono::high_resolution_clock::now();
		for (int i : tq::trange(imgs.size())){		// i is batch idx, imgs[i] is number i batch 
			size_t i_batch_size = imgs[i].size();	// actual batch size for current batch, vary at last batch
			infer(imgs[i], request, save_paths[i], i_batch_size, max_det, false, false);	// plot & log
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto durations = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		printf("\nAverage time consuming per image: %.2f ms\n", durations.count() / 1000. / float(imgs.size() * batch_size));
	}

    return 0;
};