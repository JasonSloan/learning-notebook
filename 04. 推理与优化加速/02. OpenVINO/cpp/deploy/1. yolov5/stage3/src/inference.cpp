#include <iostream>   
#include <stdio.h>

#include <string>
#include <vector>
#include <math.h>
#include <functional>                                   // std::ref()需要用这个库
#include <unistd.h>
#include <thread>                                       // 线程
#include <queue>                                        // 队列
#include <mutex>                                        // 线程锁
#include <chrono>                                       // 时间库
#include <memory>                                       // 智能指针
#include <future>                                       // future和promise都在这个库里，实现线程间数据传输
#include <condition_variable>                           // 线程通信库
#include <filesystem> 
#include <unistd.h>
#include <dirent.h>                                     // opendir和readdir包含在这里
#include <sys/stat.h>

#include "spdlog/logger.h"                              // spdlog日志相关
#include "spdlog/spdlog.h"                              // spdlog日志相关
#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include "spdlog/sinks/basic_file_sink.h"               // spdlog日志相关
#include "inference.h"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

using time_point = chrono::high_resolution_clock;
template <typename Rep, typename Period>
float micros_cast(const std::chrono::duration<Rep, Period>& d) {
    return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(d).count()) / 1000.;
}

const int anchors[3][6] = {{10, 13, 16, 30, 33, 23},
                           {30, 61, 62, 45, 59, 119},
                           {116, 90, 156, 198, 373, 326}};

string get_log_file_name(string& log_dir) {
    if (access(log_dir.c_str(), 0) != F_OK)
        mkdir(log_dir.c_str(), S_IRWXU);
    DIR* pDir = opendir(log_dir.c_str());
    struct dirent* ptr;
    vector<string> files_vector;
    while ((ptr = readdir(pDir)) != nullptr) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files_vector.push_back(ptr->d_name);
        }
    }
    closedir(pDir);
    std::sort(files_vector.begin(), files_vector.end());
    int max_num = 0;
    if (files_vector.size() != 0) {
        for (auto &file : files_vector) {
            string num_str = file.substr(0, file.find("."));
            int num = std::stoi(num_str);
            if (num > max_num)
                max_num = num;
        }
        max_num += 1;
    }

	return std::to_string(max_num);
}

struct Job{
    shared_ptr<promise<vector<Result>>> pro;        //为了实现线程间数据的传输，需要定义一个promise，由智能指针托管, pro负责将消费者消费完的数据传递给生产者
    vector<Mat> input_images;                       // 输入图像, 多batch                  
    vector<string> unique_ids;
};


class InferImpl : public InferInterface{            // 继承虚基类，从而实现load_model和destroy的隐藏
public:
    virtual ~InferImpl(){
        stop();
        spdlog::warn("Destruct instance done!");
    }

    void stop(){
        if(running_){
            running_ = false;
            cv_.notify_one();                                                   // 通知worker给break掉        
        }
        if(worker_thread_.joinable())                                           // 子线程加入     
            worker_thread_.join();
    }

    bool startup(const string& file){
        modelPath = file;
        running_ = true;                                                        // 启动后，运行状态设置为true
        promise<bool> pro;
        worker_thread_ = thread(&InferImpl::worker, this, std::ref(pro));       // 为什么要加std::ref, 因为创建新的线程的时候希望新的线程操作原始的pro,而不是复制一份新的pro
        return pro.get_future().get();	
    }

    void worker(promise<bool>& pro){
        // init log file: auto increase from 0.txt to 1.txt
        #ifndef DEBUG
            fs::remove_all(logs_dir);
        #endif
        string log_name = get_log_file_name(logs_dir);
        string log_path = logs_dir + "/" + log_name + ".txt";
        try {
            logger_ = spdlog::basic_logger_mt(log_name, log_path);
            logger_->set_level(spdlog::level::warn);
        } catch (const spdlog::spdlog_ex &ex) {
            spdlog::error("Log initialization failed: {}", ex.what());
            pro.set_value(false);
            return;
        }
        // 加载模型
        ov::Core core;
        ov::CompiledModel model = core.compile_model(modelPath, device_);
        request = model.create_infer_request();
        if(model.inputs().empty() || model.outputs().empty()){
            // failed
            pro.set_value(false);                                               // 将start_up中pro.get_future().get()的值设置为false
            logger_->error("Load model failed from path: {}!", modelPath);
            spdlog::error("Load model failed from path: {}!", modelPath);
            return;
        }
        // load success
        spdlog::info("Using device {}", device_);
        pro.set_value(true);  
        if (modelLog_) spdlog::info("Model loaded successfully from {}", modelPath);
        vector<Job> fetched_jobs;
        while(running_){
            {
                unique_lock<mutex> l(lock_);
                cv_.wait(l, [&](){return !running_ || !jobs_.empty();});        // 一直等着，cv_.wait(lock, predicate):如果 running不在运行状态 或者说 jobs_有东西 而且接收到了notify one的信号
                if(!running_) break;                                            // 如果实例被析构了，那么就结束该线程
                while(!jobs_.empty()){
                    fetched_jobs.emplace_back(std::move(jobs_.front()));        // 往里面fetched_jobs里塞东西 , std::move将对象的所有权从a转移到b 
                    jobs_.pop();                                                // 从jobs_任务队列中将当前要推理的job给pop出来 
                }                                
                l.unlock(); 
				for(auto& job : fetched_jobs){                                  // 遍历要推理的job         
                    inference(job);                                             // 调用inference执行推理
                }
                fetched_jobs.clear();
            }
        }
    }

    // forward函数是生产者, 异步返回, 在main.cpp中获取结果
    virtual std::shared_future<std::vector<Result>> forward(Input* inputs, int& n_images) override{  
        vector<Mat> input_images;
        vector<string> unique_ids;
        for (int i = 0; i < n_images; ++i){
            Mat image(inputs[i].height, inputs[i].width, CV_8UC3, inputs[i].data);
            input_images.push_back(image);
            unique_ids.push_back(inputs[i].unique_id);
        }
        batch_size = input_images.size();      
        Job job;
        job.pro.reset(new promise<vector<Result>>());
        job.input_images = input_images;
        job.unique_ids = unique_ids;

        shared_future<vector<Result>> fut = job.pro->get_future();              // get_future()并不会等待数据返回，get_future().get()才会
        {
            lock_guard<mutex> l(lock_);
            jobs_.emplace(std::move(job));                                      // 向任务队列jobs_中添加任务job
        }
        cv_.notify_one();                                                       // 通知worker线程开始工作了
        return fut;                                                             // 等待模型将推理数据返回fut，然后fut再将数据return出去
    }

    void preprocess(vector<Mat>& batched_imgs, 
                    ov::InferRequest& request, 
                    vector<vector<int>>& batched_pad_w, 
                    vector<vector<int>>& batched_pad_h, 
                    vector<float>& batched_scale_factors, 
                    size_t& batch_size, 
                    size_t& max_det){
        // set input & ouput shape for dynamic batch 
        ov::Tensor input_tensor = request.get_input_tensor();
        ov::Shape input_shape = input_tensor.get_shape();
        input_shape[0] = batch_size; // Set the batch size in the input shape
        input_tensor.set_shape(input_shape);
        size_t input_channel = input_shape[1];
        size_t input_height = input_shape[2];
        size_t input_width = input_shape[3];

        ov::Tensor output_tensor = request.get_output_tensor(0);
        ov::Shape output_shape = output_tensor.get_shape();
        auto output_shape_size = output_shape.size();
        if (output_shape_size == 2) {
            output_shape[0] = batch_size * max_det;     // 如果输出维度是2维的话, 说明已经将NMS添加到网络中了, 输出维度应该为[0, 7], 7代表 xyxy + conf + cls_id + image_idx
        } else {
            output_shape[0] = batch_size;               // 如果输出维度为3维的话, 说明NMS未添加到网络中, 输出维度应该为[0, 15120, nc+5]
        }
        output_tensor.set_shape(output_shape);

        if (inferLog_) {
            spdlog::info("Model input shape: {} x {} x {} x {}", batch_size, input_channel, input_height, input_width);
            if (output_shape_size == 2){
                spdlog::info("Model max output shape: {} x {}", output_shape[0], output_shape[1]);
            } else if (output_shape_size == 3) {
                spdlog::info("Model max output shape: {} x {} x {}", output_shape[0], output_shape[1], output_shape[2]);
            } else {
                spdlog::info("Model max output shape: {} x {} x {} x {}", output_shape[0], output_shape[1], output_shape[2],  output_shape[3]);
            }
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
            int pad_wl = round((input_width - img_new_w_unpad - 0.01) / 2);		                   
            int pad_wr = round((input_width - img_new_w_unpad + 0.01) / 2);
            int pad_ht = round((input_height - img_new_h_unpad - 0.01) / 2);
            int pad_hb = round((input_height - img_new_h_unpad + 0.01) / 2);
            cv::resize(img, img, cv::Size(img_new_w_unpad, img_new_h_unpad));
            cv::copyMakeBorder(img, img, pad_ht, pad_hb, pad_wl, pad_wr, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            batched_scale_factors.push_back(scale_factor);
            vector<int> pad_w = {pad_wl, pad_wr};
            vector<int> pad_h = {pad_ht, pad_hb};
            batched_pad_w.push_back(pad_w);
            batched_pad_h.push_back(pad_h);
        }

        // BGR-->RGB & HWC-->CHW & /255. & transfer data to input_tensor
        float* input_data_host = input_tensor.data<float>();
        float* i_input_data_host;
        size_t img_area = input_height * input_width;
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

    void postprocess_withoutNMS(vector<Result>& results, 
                    vector<vector<int>>& batched_pad_w, 
                    vector<vector<int>>& batched_pad_h, 
                    vector<float>& batched_scale_factors) {
        ov::Tensor output = request.get_output_tensor(0);
        size_t num_boxes = output.get_shape()[0];
        // xyxy + conf + cls_id + image_idx
        size_t num_dim = output.get_shape()[1];

        if (inferLog_) spdlog::info("Current batch output shape: {} x {}", num_boxes, num_dim);

        cv::Mat prob(num_boxes, num_dim, CV_32F, (float*)output.data());
        for (int i = 0; i < num_boxes; i++) {
            float conf = prob.at<float>(i, 4);
            int image_idx = static_cast<int>(prob.at<float>(i, 6));
            int class_id = static_cast<int>(prob.at<float>(i, 5));

            vector<int> pad_w = batched_pad_w[image_idx];
            vector<int> pad_h = batched_pad_h[image_idx];
            float scale_factor = batched_scale_factors[image_idx];
            int predx1 = std::round((prob.at<float>(i, 0) - float(pad_w[0])) / scale_factor);
            int predy1 = std::round((prob.at<float>(i, 1) - float(pad_h[0])) / scale_factor);
            int predx2 = std::round((prob.at<float>(i, 2) - float(pad_w[1])) / scale_factor);
            int predy2 = std::round((prob.at<float>(i, 3) - float(pad_h[1])) / scale_factor);
            
            vector<int> cords = {predx1, predy1, predx2, predy2};
            results[image_idx].boxes.emplace_back(cords);
            results[image_idx].labels.emplace_back(class_id);
            results[image_idx].scores.emplace_back(conf);
            if (inferLog_)
                spdlog::info("image_idx: {}, class_id: {}, conf: {}, xyxy: {} {} {} {}", image_idx, class_id, conf, predx1, predy1, predx2, predy2);
        }
    }

    void postprocess_withNMS(int& output_shape_size,
                    vector<Result>& results, 
                    vector<vector<int>>& batched_pad_w, 
                    vector<vector<int>>& batched_pad_h, 
                    vector<float>& batched_scale_factors) {

        for (int i_img = 0; i_img < batch_size; i_img++){
            vector<vector<float>> bboxes;                                      // 初始化变量bboxes:[[x1, y1, x2, y2, conf, label], [x1, y1, x2, y2, conf, label]...]
            if (output_shape_size == 3){
                decode_boxes_1output(i_img, bboxes, batched_pad_w, batched_pad_h, batched_scale_factors);
            } else {
                decode_boxes_3output(i_img, bboxes, batched_pad_w, batched_pad_h, batched_scale_factors);
            }
            
            if (inferLog_) spdlog::info("Decoded bboxes.size = {}", bboxes.size());

            // nms非极大抑制
            // 通过比较索引为5(confidence)的值来将bboxes所有的框排序
            std::sort(bboxes.begin(), bboxes.end(), [](vector<float> &a, vector<float> &b)
                    { return a[5] > b[5]; });
            std::vector<bool> remove_flags(bboxes.size()); // 设置一个vector，存储是否保留bbox的flags
            // 定义一个lambda的iou函数
            auto iou = [](const vector<float> &a, const vector<float> &b)
            {
                float cross_left = std::max(a[0], b[0]);
                float cross_top = std::max(a[1], b[1]);
                float cross_right = std::min(a[2], b[2]);
                float cross_bottom = std::min(a[3], b[3]);

                float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
                float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
                if (cross_area == 0 || union_area == 0)
                    return 0.0f;
                return cross_area / union_area;
            };

            for (int i = 0; i < bboxes.size(); ++i){
                if (remove_flags[i])
                    continue;                                   // 如果已经被标记为需要移除，则continue

                auto &ibox = bboxes[i];                         // 获得第i个box
                auto float2int = [] (float x) {return static_cast<int>(round(x));};
                vector<int> _ibox = {float2int(ibox[0]), float2int(ibox[1]), float2int(ibox[2]), float2int(ibox[3])};
                results[i_img].boxes.emplace_back(_ibox);               // 将该box放入box_result中，emplace_back和push_back基本一样，区别在于emplace_back是inplace操作
                results[i_img].labels.emplace_back(int(ibox[4]));
                results[i_img].scores.emplace_back(ibox[5]);
                for (int j = i + 1; j < bboxes.size(); ++j){    // 遍历剩余框，与box_result中的框做iou
                    if (remove_flags[j])
                        continue;                               // 如果已经被标记为需要移除，则continue

                    auto &jbox = bboxes[j];                     // 获得第j个box
                    if (ibox[4] == jbox[4]){ 
                        // class matched
                        if (iou(ibox, jbox) >= nms_threshold)   // iou值大于阈值，将该框标记为需要remove
                            remove_flags[j] = true;
                    }
                }
            }
            if (inferLog_) spdlog::info("box_result.size = {}", results[i_img].boxes.size());
        }
    }

    void decode_boxes_1output(int& i_img, 
                            vector<vector<float>>& bboxes, 
                            vector<vector<int>>& batched_pad_w, 
                            vector<vector<int>>& batched_pad_h, 
                            vector<float>& batched_scale_factors){
        ov::Tensor output = request.get_output_tensor(0);
        size_t output_numbox = output.get_shape()[1];
        size_t output_numprob = output.get_shape()[2];
        size_t offset_per_image = output_numbox * output_numprob;
        // decode and filter boxes by conf_thre
        float* output_data_host = (float*)output.data();                   // fetch index 0 because there is only one output   
        int num_classes = output_numprob - 5;
        for (int i = 0; i < output_numbox; ++i) {
            float *ptr = output_data_host + offset_per_image * i_img + i * output_numprob;             // 每次偏移output_numprob
            float objness = ptr[4];                                         // 获得置信度
            if (objness < confidence_threshold)
                continue;

            float *pclass = ptr + 5;                                        // 获得类别开始的地址
            int label = max_element(pclass, pclass + num_classes) - pclass; // 获得概率最大的类别
            float prob = pclass[label];                                     // 获得类别概率最大的概率值
            float confidence = prob * objness;                              // 计算后验概率
            if (confidence < confidence_threshold)
                continue;

            // xywh
            float cx = ptr[0];
            float cy = ptr[1];
            float width = ptr[2];
            float height = ptr[3];

            // xyxy
            float left = cx - width * 0.5;
            float top = cy - height * 0.5;
            float right = cx + width * 0.5;
            float bottom = cy + height * 0.5;

            // the box cords on the origional image
            float image_base_left = (left - batched_pad_w[i_img][0]) / batched_scale_factors[i_img];                                                                     // x1
            float image_base_right = (right - batched_pad_w[i_img][1]) / batched_scale_factors[i_img];                                                                   // x2
            float image_base_top = (top - batched_pad_h[i_img][0]) / batched_scale_factors[i_img];                                                                       // y1
            float image_base_bottom = (bottom - batched_pad_h[i_img][1]) / batched_scale_factors[i_img];
            bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)label, confidence}); // 放进bboxes中
        }
    }
    
    void decode_boxes_3output(int& i_img,
                    vector<vector<float>>& bboxes,
                    vector<vector<int>>& batched_pad_w, 
                    vector<vector<int>>& batched_pad_h, 
                    vector<float>& batched_scale_factors){
        int anchor_size = sizeof(anchors[0]) / sizeof(int) / 2;
        vector<ov::Tensor> outputs;
        vector<ov::Shape> output_shapes;
        vector<size_t> offsets;
        for (int i = 0; i < 3; ++i) {
            ov::Tensor i_output = request.get_output_tensor(i);
            ov::Shape i_output_shape = i_output.get_shape();
            size_t i_offset =  i_output_shape[1] * i_output_shape[2] * i_output_shape[3];
            outputs.push_back(i_output);
            output_shapes.push_back(i_output_shape);
            offsets.push_back(i_offset);
        }

        // decode boxes
        int prob_box_size = output_shapes[0][1] / anchor_size;
        int nc = prob_box_size - 5;
        // iter every output
        for (int ifeat = 0; ifeat < 3; ++ifeat) {
            int grid_h = output_shapes[ifeat][2];
            int grid_w = output_shapes[ifeat][3];
            int grid_len = grid_w * grid_h;
            int stride = request.get_input_tensor().get_shape()[3] / grid_w;
            float* output_data_host = (float*)outputs[ifeat].data() + i_img * offsets[ifeat];
            // iter every anchor
            for (int a = 0; a < anchor_size; a++){
                // iter every h grid                        
                for (int i = 0; i < grid_h; i++){
                    // iter every w grid
                    for (int j = 0; j < grid_w; j++){
                        // 想象成一个魔方, j是w维度, i是h维度, a是深度
                        // xyxy + conf + cls_prob
                        float box_confidence = output_data_host[(prob_box_size * a + 4) * grid_len + i * grid_w + j];
                        if (box_confidence > confidence_threshold){
                            int offset = (prob_box_size * a) * grid_len + i * grid_w + j;
                            float *in_ptr = output_data_host + offset;
                            float box_x = in_ptr[0] * 2.0 - 0.5;
                            float box_y = in_ptr[grid_len] * 2.0 - 0.5;
                            float box_w = in_ptr[2 * grid_len] * 2.0;
                            float box_h = in_ptr[3 * grid_len] * 2.0;
                            box_x = (box_x + j) * (float)stride;
                            box_y = (box_y + i) * (float)stride;
                            box_w = box_w * box_w * (float)anchors[ifeat][a * 2];
                            box_h = box_h * box_h * (float)anchors[ifeat][a * 2 + 1];
                            float box_x1 = box_x - (box_w / 2.0);
                            float box_y1 = box_y - (box_h / 2.0);
                            float box_x2 = box_x + (box_w / 2.0);
                            float box_y2 = box_y + (box_h / 2.0);
                            float maxClassProbs = in_ptr[5 * grid_len];
                            int maxClassId = 0;
                            for (int t = 1; t < nc; ++t){
                                float prob = in_ptr[(5 + t) * grid_len];
                                if (prob > maxClassProbs){
                                    maxClassId = t;
                                    maxClassProbs = prob;
                                }
                            }
                            if (maxClassProbs > confidence_threshold){
                                float confidence = maxClassProbs * box_confidence;
                                float image_base_left = (box_x1 - batched_pad_w[i_img][0]) / batched_scale_factors[i_img];                                                                     // x1
                                float image_base_right = (box_x2 - batched_pad_w[i_img][1]) / batched_scale_factors[i_img];                                                                   // x2
                                float image_base_top = (box_y1 - batched_pad_h[i_img][0]) / batched_scale_factors[i_img];                                                                       // y1
                                float image_base_bottom = (box_y2 - batched_pad_h[i_img][1]) / batched_scale_factors[i_img];  
                                bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)maxClassId, confidence});
                            }
                        }
                    }
                }
            }
        }
    }

    void postprocess_entry(vector<Result>& results, 
                    ov::InferRequest & request, 				
                    vector<vector<int>>& batched_pad_w, 
                    vector<vector<int>>& batched_pad_h, 
                    vector<float>& batched_scale_factors, 
                    size_t& batch_size) {
        ov::Tensor output_tensor = request.get_output_tensor(0);
        ov::Shape output_shape = output_tensor.get_shape();
        int output_shape_size = output_shape.size();
        if (output_shape_size == 2) {
            postprocess_withoutNMS(results, batched_pad_w, batched_pad_h, batched_scale_factors);       // use this if NMS is integrated into the network
        } else {
            postprocess_withNMS(output_shape_size, results, batched_pad_w, batched_pad_h, batched_scale_factors);          // use this if NMS is not integrated into the network
        }
    }

    void inference(Job& job){
        // Initialize a dummy Result object
        Result dummy;
        dummy.unique_id = "-1";  
        dummy.boxes = {};        
        dummy.labels = {};    
        dummy.scores = {};  
        std::vector<Result> results(batch_size, dummy);
        // fetch data
        vector<Mat> batched_imgs = job.input_images;
        vector<string> unique_ids = job.unique_ids;
        // initialize time
        auto start = time_point::now();
        auto stop = time_point::now();

        // preprocess 
        start = time_point::now();
        vector<vector<int>> batched_pad_w, batched_pad_h;
        vector<float> batched_scale_factors;
        preprocess(batched_imgs, request, batched_pad_w, batched_pad_h, batched_scale_factors, batch_size, max_det);
        stop = time_point::now();
        InferImpl::records[0].push_back(micros_cast(stop - start));

        // infer
        start = time_point::now();
        do_infer(request);
        stop = time_point::now();
        InferImpl::records[1].push_back(micros_cast(stop - start));

        // postprocess
        start = time_point::now();
        postprocess_entry(results, request, batched_pad_w, batched_pad_h, batched_scale_factors, batch_size);
        for (int i = 0; i < batch_size; ++i) {
            results[i].unique_id = unique_ids[i];           // attach each id to results
        }
        stop = time_point::now();
        InferImpl::records[2].push_back(micros_cast(stop - start));

        // return results
        job.pro->set_value(results);
    }

private:
    // 可调数据
    string modelPath;                                           // 模型路径
    size_t batch_size;
    size_t max_det{100};                                        
    string device_;
    float confidence_threshold{0.5};
    float nms_threshold{0.6};
    // 多线程有关
    atomic<bool> running_{false};                               // 如果InferImpl类析构，那么开启的子线程也要break
    thread worker_thread_;
    queue<Job> jobs_;                                           // 任务队列
    mutex lock_;                                                // 负责任务队列线程安全的锁
    condition_variable cv_;                                     // 线程通信函数
    size_t pool_size{1};
    // 模型初始化有关           
    ov::InferRequest request;
    //日志相关
    bool modelLog_{true};                                             // 模型加载时是否打印日志在控制台
    bool inferLog_{false};                                             // 模型推理时是否打印日志在控制台
    std::shared_ptr<spdlog::logger> logger_;                    // logger_负责日志文件, 记录一些错误日志
    string logs_dir{"logs"};                                    // 日志文件存放的文件夹   
    // 计时相关
    static vector<vector<float>> records;                       // 计时相关: 静态成员变量声明
};

vector<vector<float>> InferImpl::records(3);                    // 计时相关: 静态成员变量定义, 长度为3

shared_ptr<InferInterface> create_infer(std::string &modelPath){
    shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(modelPath)){
        instance.reset();                                                     
    }
    return instance;
};