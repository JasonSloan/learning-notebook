官网notebook各种案例:https://github.com/openvinotoolkit/openvino_notebooks

ps:第220项YOLOv5量化int8

# 一. 安装

```C++
pip install openvino-dev[pytorch,onnx]
```

# 二. OpenVINO流程

![](assets/flow.png)

# 三. 查看可用设备

```python
from openvino.runtime import Core
# 初始化 OpenVINO Runtime with Core()
ie = Core()               
# 查看可用设备类型：CPU还是GPU
devices = ie.available_devices
# 查看可用设备型号：酷睿i5还是i7...
for device in devices:
    device_name = ie.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")
```



# 四. 模型加载

```python
# OpenVINO IR模型由两部分组成：.xml文件和.bin文件；.xml文件包含模型拓扑结构，.bin文件包含模型权重。
# 这两种文件可以通过Model Optimizer tool获得。这两种文件必须同名（后缀名不同），且放在同一文件夹下。


#=============方式一：加载IR类型模型=============  
from openvino.runtime import Core
# 初始化 OpenVINO Runtime with Core()
ie = Core()   
# 指定xml文件，bin文件同名且与xml文件同目录下
classification_model_xml = "model/classification.xml"
# 读取模型
model = ie.read_model(model=classification_model_xml)
# 编译模型
compiled_model = ie.compile_model(model=model, device_name="CPU")


#=============方式二：加载ONNX类型模型============= 
from openvino.runtime import Core
# 初始化 OpenVINO Runtime with Core()
ie = Core() 
onnx_model_path = "model/segmentation.onnx"
model_onnx = ie.read_model(model=onnx_model_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")


#=============方式三：加载PaddlePaddle类型模型============= 
from openvino.runtime import Core
ie = Core()
paddle_model_path = "model/inference.pdmodel"
model_paddle = ie.read_model(model=paddle_model_path)
compiled_model_paddle = ie.compile_model(model=model_paddle, device_name="CPU")


#=============方式四、加载TensorFlow类型模型============= 
from openvino.runtime import Core
ie = Core()
tf_model_path = "model/classification.pb"
model_tf = ie.read_model(model=tf_model_path)
compiled_model_tf = ie.compile_model(model=model_tf, device_name="CPU")
```

# 五. 模型转换

```python
#=============1，ONNX转换为IR模型=============
from openvino.runtime import Core
from openvino.runtime import serialize
ie = Core()
onnx_model_path = "model/segmentation.onnx"
model_onnx = ie.read_model(model=onnx_model_path)
serialize(model_onnx, xml_path="model/exported_onnx_model.xml")


#=============2，PaddlePaddle转换为IR模型=============
from openvino.runtime import Core
from openvino.runtime import serialize
ie = Core()
paddle_model_path = "model/inference.pdmodel"
model_paddle = ie.read_model(model=paddle_model_path)
serialize(model_paddle, xml_path="model/exported_paddle_model.xml")


#=============3，TensorFlow转换为IR模型=============
from openvino.runtime import Core
from openvino.runtime import serialize
ie = Core()
tf_model_path = "model/classification.pb"
model_tf = ie.read_model(model=tf_model_path)
serialize(model_tf, xml_path="model/exported_tf_model.xml")


#=============4，PyTorch转换为IR模型=============
方式一：
# 使用torch.onnx.export将torch模型转换为onnx模型
# 将onnx模型放在与mo.py同级目录下
# 执行命令：
python3 <OpenVINO_INSTALL_DIR>/mo.py \
--input_model <path_to_input_model> \
--compress_to_fp16 \
--output_dir <path_to_output_directory> \
--model_name <model_name>
方式二：
from openvino.tools import mo
from openvino.runtime import serialize
onnx_path = f"{MODEL_PATH}/{MODEL_NAME}.onnx"
# fp32 IR model
fp32_path = f"{MODEL_PATH}/FP32_openvino_model/{MODEL_NAME}_fp32"
output_path = fp32_path + ".xml"
print(f"Export ONNX to OpenVINO FP32 IR to: {output_path}")
model = mo.convert_model(onnx_path)
serialize(model, output_path)
# fp16 IR model
fp16_path = f"{MODEL_PATH}/FP16_openvino_model/{MODEL_NAME}_fp16"
output_path = fp16_path + ".xml"
print(f"Export ONNX to OpenVINO FP16 IR to: {output_path}")
model = mo.convert_model(onnx_path, data_type="FP16", compress_to_fp16=True)
serialize(model, output_path)
```

# 六. 获取模型输入输出属性

```python
from openvino.runtime import Core
ie = Core()
classification_model_xml = "model/classification.xml"
model = ie.read_model(model=classification_model_xml)
# 获取输入模型属性方式一：输入名称、输入形状、输入数据类型（fp32 etc）
model.inputs    # [<Output: names[input:0, input] shape[1,3,224,224] type: f32>]
# 获取输入模型属性方式二：输入名称、输入形状、输入数据类型（fp32 etc）
model.input(0).any_name          # 'input'
model.input(0).element_type      # input precision: <Type: 'float32'>
model.input(0).shape             # input shape: [1,3,224,224]
```

# 七. 模型加载并推理全流程

```python
# 1，模型加载
from openvino.runtime import Core
ie = Core()
classification_model_xml = "model/classification.xml"
model = ie.read_model(model=classification_model_xml)
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# 2，图像加载、resize、转换成NCHW
import cv2
image_filename = "../data/image/coco_hollywood.jpg"
image = cv2.imread(image_filename)
N, C, H, W = input_layer.shape
# OpenCV resize expects the destination size as (width, height).
resized_image = cv2.resize(src=image, dsize=(W, H))
import numpy as np
input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)


# 3，推理
# for single input models only
result = compiled_model(input_data)[output_layer]

# for multiple inputs in a list
result = compiled_model([input_data])[output_layer]

# or using a dictionary, where the key is input tensor name or index
result = compiled_model({input_layer.any_name: input_data})[output_layer]

```

