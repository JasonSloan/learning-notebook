#include <string>
#include <vector>
#include <chrono>
#include <stdio.h>
#include <dirent.h>     // opendir和readdir包含在这里

#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include "inference.hpp"

using namespace std;
using namespace cv;

const vector<Scalar> COLORS = {
	{255, 0, 0},
	{0, 255, 0},
	{0, 0, 255},
	{0, 255, 255}
};

const string LABELS[] = {"head", "helmet", "person", "lookout"};

void print_avaliable_devices() {
	ov::Core core;
	vector<string> availableDevices = core.get_available_devices();
	for (int i = 0; i < availableDevices.size(); i++) {
		printf("supported device name : %s \n", availableDevices[i].c_str());
	}
}

int listdir(string& input,  vector<string>& files_vector) {
    DIR* pDir = opendir(input.c_str());
    if (!pDir) {
        cerr << "Error opening directory: " << strerror(errno) << endl;
        return -1;
    }
    struct dirent* ptr;
    while ((ptr = readdir(pDir)) != nullptr) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files_vector.push_back(input + "/" + ptr->d_name);
        }
    }
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

string getFileName(const std::string& file_path, bool with_ext=true){
	int index = file_path.find_last_of('/');
	if (index < 0)
		index = file_path.find_last_of('\\');
    std::string tmp = file_path.substr(index + 1);
    if (with_ext)
        return tmp;
    std::string img_name = tmp.substr(0, tmp.find_last_of('.'));
    return img_name;
}

void draw_rectangles(vector<Result>& results, vector<Mat>& im0s, vector<string>& save_paths){
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
			cv::imwrite(save_paths[i], im0);
		}
	}
}