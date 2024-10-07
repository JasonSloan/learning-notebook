import torch



from pytorch_quantization import quant_modules
from yolov7.models.yolo import Model
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib


def load_yolov7_model(weight, device='cpu'):
    ckpt = torch.load(weight, map_location=device)
    model = Model("yolov7/cfg/training/yolov7.yaml", ch=3, nc=80).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    return model

from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from absl import logging as quant_logging
# intput: Max ==> Histogram
def initialize():
    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    quant_logging.set_verbosity(quant_logging.ERROR)

def prepare_model(weight, device):
    # quant_modules.initialize()
    initialize()
    model = load_yolov7_model(weight, device)
    model.float()
    model.eval()
    with torch.no_grad():
        model.fuse()  # conv bn 进行层的合并, 加速
    return model


def transfer_torch_to_quantization(nn_instance, quant_mudule):
    
    quant_instance = quant_mudule.__new__(quant_mudule)

    for k, val in vars(nn_instance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        # 返回两个QuantDescriptor的实例    self.__class__是quant_instance的类, EX: QuantConv2d
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance




# 递归函数
def torch_module_find_quant_module(module, module_dict, prefix=''):
    for name in module._modules:
        submodule = module._modules[name]
        path =  name if prefix == '' else prefix + '.' + name
        torch_module_find_quant_module(submodule, module_dict, prefix=path)

        submodule_id = id(type(submodule))
        if submodule_id in module_dict:
            # 转换
            module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])




def replace_to_quantization_model(model):

    module_dict = {}

    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod

    torch_module_find_quant_module(model, module_dict)





import collections
from yolov7.utils.datasets import create_dataloader
def prepare_val_dataset(cocodir, batch_size=4):
    dataloader = create_dataloader(
        f"{cocodir}/val2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=False, hyp=None, rect=True, cache=False, stride=32,pad=0.5, image_weights=False
    )[0]
    return dataloader

import yaml
def prepare_train_dataset(cocodir, batch_size=4):
    with open("yolov7/data/hyp.scratch.p5.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    dataloader = create_dataloader(
        f"{cocodir}/train2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=True, hyp=hyp, rect=True, cache=False, stride=32,pad=0, image_weights=False
    )[0]



    return dataloader

import yolov7.test as test
from pathlib import Path
import os
def evaluate_coco(model, loader, save_dir='.', conf_thres=0.001, iou_thres=0.65):
    
    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    return test.test(
        "yolov7/data/coco.yaml",
        save_dir=Path(save_dir),
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        model=model,
        dataloader=loader,
        is_coco=True,
        plots=False,
        half_precision=True,
        save_json=False
    )[0][3]


def collect_stats(model, data_loader, device, num_batch=200):
    model.eval()

    # 开启校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # test
    with torch.no_grad():
        for i, datas in enumerate(data_loader):
            imgs = datas[0].to(device, non_blocking=True).float()/255.0
            model(imgs)

            if i >= num_batch:
                break
    # 关闭校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()



def compute_amax(model, **kwargs):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
                module._amax = module._amax.to(device)


def calibrate_model(model, dataloader, device):
    # 收集信息
    collect_stats(model, dataloader, device)
    # 获取动态范围,计算amax值,scale值
    compute_amax(model, method='mse')


def export_ptq(model, save_file, device, dynamic_batch=False):

    input_dummy = torch.randn(1, 3, 640, 640, device=device)

    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, input_dummy, save_file, opset_version=13,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}} if dynamic_batch else None,
        )

    quant_nn.TensorQuantizer.use_fb_fake_quant = False


def have_quantizer(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True


class disable_quantization:
    def __init__(self, model):
        self.model  = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    def __init__(self, model):
        self.model  = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(False)


import json
class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []

    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)


def sensitive_analysis(model, val_loader):
    save_file = "sensitive_analysis.json"
    summary = SummaryTool(save_file)

    # for 循环所有层
    print("Sensitive analysis by each layer .....")
    for i in range(0, len(model.model)):
        layer = model.model[i]
        # 判断该层是否是quantizer量化层
        if have_quantizer(layer):
            # 关闭该层的量化
            disable_quantization(layer).apply()
            ap = evaluate_coco(model, val_loader)  # 计算mAP
            print(f"layer {i} ap: {ap}")
            # save ap
            summary.append([ap, f"model.{i}"])
            # 重启该层的量化
            enable_quantization(layer).apply()
        else:
            print(f"ignore model.{i} because it is {type(layer)}")

    summary = sorted(summary.data, key=lambda x: x[0], reverse=True)  # 由大到小
    print("Sensitive summary: ")
    for n, (ap, name) in enumerate(summary[:10]):
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")


if __name__ == "__main__":

    weight = "yolov7.pt"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    print("Evaluate Dataset...")
    cocodir = "dataset/coco2017"
    val_dataloader = prepare_val_dataset(cocodir)
    train_dataloader = prepare_train_dataset(cocodir)

    # 加载pth模型
    # pth_model = load_yolov7_model(weight, device)
    # pth模型验证
    # print("Evaluate Origin...")
    # pth_ap = evaluate_coco(pth_model, dataloader)

    # 获取伪量化模型(手动initial(), 手动插入QDQ)
    model = prepare_model(weight, device)
    replace_to_quantization_model(model)


    # 模型标定,使用训练数据集
    calibrate_model(model, train_dataloader, device)

    

    # 敏感层分析
    '''
    1. for循环module的每一个quantizer量化层
    2. 只关闭该量化层,其余量化层不变
    3. 进行模型精度的验证, 使用evaluate_coco(), 并保存精度值
    4. 验证结束,需要重启该层的量化
    5. for循环结束, 获得所有层的精度值
    6. 获取前10个对精度影响最大的层,并打印出来
    '''
    sensitive_analysis(model, val_dataloader)

    # print("Export PTQ...")
    # export_ptq(model, "ptq_yolov7.onnx", device)



    # ptq模型验证
    # print("Evaluate Histogram PTQ...")
    # ptq_ap = evaluate_coco(model, dataloader)




