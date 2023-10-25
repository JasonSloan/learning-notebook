#include <stdio.h>
// #include "opencv2/opencv.hpp"
#include "inference.hpp"

int main(){
    // 动态库调用使用这段，需要指定../workspace/
    std::string modelPath = "../workspace/engine.trtmodel";
    auto infer = create_infer(modelPath); 
    if (infer == nullptr){           // 调用者只需要判断指针是否为空即可
        printf("load model failed.\n");
        return -1;
    }
    std::string imagePath = "../workspace/bytes_image.bin";
    int height = 50;
    int width = 80;    
    std::vector<uint8_t> output_image_bytes;
    output_image_bytes = infer->forward(imagePath, width, height); // infer只能调用forward，也只能看见forward
    return 0;
}