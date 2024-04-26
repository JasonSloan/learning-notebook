#include <iostream>   
#include <stdio.h>
#include <string>
#include <vector>
#include <math.h>
#include <functional>                                                       // std::ref()需要用这个库
#include <unistd.h>
#include <thread>                                                           // 线程
#include <queue>                                                            // 队列
#include <mutex>                                                            // 线程锁
#include <chrono>                                                           // 时间库
#include <memory>                                                           // 智能指针
#include <future>                                                           // future和promise都在这个库里，实现线程间数据传输
#include <condition_variable>                                               // 线程通信库

#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include "inference.hpp"

using namespace std;
using namespace cv;


struct Job{
    shared_ptr<promise<vector<Result>>> pro;        //为了实现线程间数据的传输，需要定义一个promise，由智能指针托管, pro负责将消费者消费完的数据传递给生产者
    vector<Mat> input_images;                       // 输入图像, 多batch                  
    bool log;                                       // 是否打印日志
};


class InferImpl : public InferInterface{            // 继承虚基类，从而实现load_model和destroy的隐藏
public:
    virtual ~InferImpl(){
        stop();
        printf("Destruct instance done!\n");
    }

    void stop(){
        if(running_){
            running_ = false;
            cv_.notify_one();                                                   // 通知worker给break掉        
        }
        if(worker_thread_.joinable())                                           // 子线程加入     
            worker_thread_.join();
    }

    bool startup(const string& file, 
                 const size_t& md){
        modelPath = file;
        max_det = md;
        running_ = true;                                                        // 启动后，运行状态设置为true
        promise<bool> pro;
        worker_thread_ = thread(&InferImpl::worker, this, std::ref(pro));       // 为什么要加std::ref, 因为创建新的线程的时候希望新的线程操作原始的pro,而不是复制一份新的pro
        return pro.get_future().get();	
    }

    void worker(promise<bool>& pro){
        // 加载模型
        ov::Core core;
        ov::CompiledModel model = core.compile_model(modelPath, "AUTO");
        request = model.create_infer_request();
        if(model.inputs().empty() || model.outputs().empty()){
            // failed
            pro.set_value(false);                                               // 将start_up中pro.get_future().get()的值设置为false
            printf("Model loaded failuer: %s\n", modelPath.c_str());
            return;
        }
        // load success
        pro.set_value(true);  
        printf("Model loaded successfully: %s\n", modelPath.c_str());
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
                    auto start_time = std::chrono::high_resolution_clock::now();                            // [&]{inference(job); return;}  
                    inference(job);                                             // 调用inference执行推理
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                    // printf("Total time consumed: %ld ms\n", duration.count());
                }
                fetched_jobs.clear();
            }
        }
    }

    // 实际上forward函数是生产者, 异步返回, 在main.cpp中获取结果
    virtual std::shared_future<std::vector<Result>> forward(std::vector<cv::Mat> input_images, 
		                           bool log=false) override{  
    // virtual std::vector<Result> forward(std::vector<cv::Mat> input_images, 
	// 	                           bool log=false) override{  
        batch_size = input_images.size();      
        Job job;
        job.pro.reset(new promise<vector<Result>>());
        job.input_images = input_images;
        job.log = log;

        shared_future<vector<Result>> fut = job.pro->get_future();              // get_future()并不会等待数据返回，get_future().get()才会
        {
            lock_guard<mutex> l(lock_);
            jobs_.emplace(std::move(job));                                      // 向任务队列jobs_中添加任务job
        }
        cv_.notify_one();                                                       // 通知worker线程开始工作了
        return fut;
        // return fut.get();                                                             // 等待模型将推理数据返回fut，然后fut再将数据return出去
    }

    void preprocess(vector<Mat>& batched_imgs, 
                    ov::InferRequest& request, 
                    vector<vector<int>>& batched_pad_w, 
                    vector<vector<int>>& batched_pad_h, 
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

    void postprocess(vector<Result>& results, 
                    ov::InferRequest & request, 				
                    vector<vector<int>>& batched_pad_w, 
                    vector<vector<int>>& batched_pad_h, 
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
            if (log) {
                printf("image_idx: %d, class_id: %d, conf: %.2f, xyxy: %d %d %d %d\n", image_idx, class_id, conf, predx1, predy1, predx2, predy2);
            }
        }
    }

    void print_consuming_time(std::vector<std::chrono::microseconds>& durations){
        printf("Consuming time: %.2f ms preprocess, %.2f ms infer, %.2f ms postprocess\n", 
                durations[0].count() / 1000.0, durations[1].count() / 1000.0, durations[2].count() / 1000.0);
        durations.clear();
    }
    
    void inference(Job& job){
        // fetch data
        vector<Mat> batched_imgs = job.input_images;
        bool log = job.log;

        // initialize time
        std::vector<std::chrono::microseconds> durations;
        auto start = std::chrono::high_resolution_clock::now();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        // preprocess 
        start = std::chrono::high_resolution_clock::now();
        vector<vector<int>> batched_pad_w, batched_pad_h;
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

        // print time consuming
        if (log == true){
            print_consuming_time(durations);
        }

        // return results
        job.pro->set_value(results);
    }

private:
    // 可调数据
    string modelPath;                                           // 模型路径
    size_t batch_size;
    size_t max_det;
    // 多线程有关
    atomic<bool> running_{false};                               // 如果InferImpl类析构，那么开启的子线程也要break
    thread worker_thread_;
    queue<Job> jobs_;                                           // 任务队列
    mutex lock_;                                                // 负责任务队列线程安全的锁
    condition_variable cv_;                                     // 线程通信函数
    size_t pool_size{1};
    // 模型初始化有关           
    ov::InferRequest request;
    
};

shared_ptr<InferInterface> create_infer(const string &file, 
                                        size_t max_det){                        // 返回的指针向虚基类转化
    shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(file, max_det)){
        instance.reset();                                                       // 如果模型加载失败，instance要reset成空指针
    }
    return instance;
};