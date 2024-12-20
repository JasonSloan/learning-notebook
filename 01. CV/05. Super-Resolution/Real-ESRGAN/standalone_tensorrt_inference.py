import cv2
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA context
import time


class ESRGAN():
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float16)
        img = img / 255.
        img = np.ascontiguousarray(img.transpose([2, 0, 1])[None])
        return img

    def post_process(self, output_img):
        output_img = np.clip(output_img.squeeze(), 0, 1)
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        output = (output_img * 255.0).round().astype(np.uint8)
        return output

    def inference(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        input_data = self.preprocess(img)
        output_hw = [x * 4 for x in input_data.shape[2:]]
        output_shape = [1, 3, *output_hw]
        output_data = np.empty(output_shape, dtype=np.float16)
        d_input = cuda.mem_alloc(input_data.nbytes)
        d_output = cuda.mem_alloc(output_data.nbytes)
        cuda.memcpy_htod(d_input, input_data)
        self.context.execute_v2(bindings=[int(d_input), int(d_output)])
        cuda.memcpy_dtoh(output_data, d_output)
        d_input.free()
        d_output.free()
        output = self.post_process(output_data)
        return output


if __name__ == '__main__':
    # TODO: 已将模型导出为onnx，又将onnx导出为engine，使用engine推理报错（因为指定了动态宽高，难道是因为在导出engine的时候动态宽高设置不正确？）
    # 已查验，onnx没问题是动态宽高
    engine_path = '/root/rrdb.engine'
    img_path = 'inputs/00003.png'
    esrgan = ESRGAN(engine_path)
    esrgan.inference(img_path)
