"""
新增的代码文件:
ultralytics/nn/modules/block_pruned.py: 新增Bottleneckpruned, C2fpruned, SPPFpruned 
"""
import os
import sys
from pathlib import Path

import yaml
import argparse

import torch
import torch.nn as nn
from ultralytics.utils import colorstr
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules import Conv, Concat, Detect

from ultralytics.nn.modules.block_pruned import Bottleneckpruned, C2fpruned, SPPFpruned

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def main(opt):
    weights, prune_ratio, cfg, model_size = opt.weights, opt.prune_ratio, opt.cfg, opt.model_size
    model = AutoBackend(weights, fuse=False)
    model.eval()
    # =========================================step1=========================================
    """
    遍历所有module:
        将module中的所有bn层按照{名字: bn层}的形式存在一个字典中;
        同时将忽略剪枝的bn层的名字存在一个列表中(具有残差结构的位置);
    """
    # 聚集所有的bn层和忽略剪枝的bn层
    model_list = {}
    ignore_bn_list = []
    for name, module in model.model.named_modules():
        if isinstance(module, Bottleneck):
            if module.add:
                ignore_bn_list.append(f"{name[:-4]}.cv1.bn")
                ignore_bn_list.append(f"{name}.cv2.bn")
        if isinstance(module, nn.BatchNorm2d):
            model_list[name] = module     
    # 检查下忽略剪枝的bn层是否有命名错误的        
    for ignore_bn_name in ignore_bn_list:
        assert ignore_bn_name in model_list.keys()
    # =========================================step2========================================= 
    """按照忽略列表过滤已存储的bn层的字典"""
    # 按照ignore_bn_list过滤model_list
    model_list = {k : v for k, v in model_list.items() if k not in ignore_bn_list}
    # =========================================step3=========================================
    """遍历剩余的bn层, 将bn层的gamma值拿出来, 按照顺序依次放入同一个列表中(即将要剪枝的bn层的**gamma的绝对值**全都gather起来)"""
    bn_weights = []
    for name, module in model_list.items():
        bn_weights.extend(module.weight.data.abs().cpu().clone().tolist())
    # =========================================step4========================================= 
    """将step3中的列表按照数值大小排序得到一个新的已排序好的数组, 准备计算阈值"""
    sorted_bn = torch.sort(torch.tensor(bn_weights))[0]
    # ========================================step5=========================================
    """
    遍历剩余的bn层, 计算每一个bn层的gamma的绝对值的最大值(即当前层gamma的最大值), 存入一个列表中
    计算step5中的那个列表中所有"层最大值"的最小值(也就是所有局部最大值的最小值), 剪枝的阈值不能超过该值, 如果超过该值, 
    那么就存在可能某一层整体被剪掉
    按照剪枝比率计算剪枝阈值: 即按比率计算step4中已排序好的数组对应索引, 按照step4中已排序好的数组取索引对应的值即为阈值
    """
    highest_thre = []
    for name, module in model_list.items():
        highest_thre.append(module.weight.data.abs().cpu().clone().max())
    highest_thre = min(highest_thre)
    percent_limit = (sorted_bn == highest_thre).nonzero()[0, 0].item() / len(sorted_bn)
    thre = sorted_bn[int(len(sorted_bn) * prune_ratio)]
    print(colorstr('Attention:'))
    print(f'Suggested Gamma threshold should be less than {highest_thre:.4f}, yours is {thre:.4f}')
    print(f'The corresponding prune ratio should be less than {percent_limit:.3f}, yours is {prune_ratio:.3f}')
    # ========================================step6=========================================
    """将模型配置文件重新保存为一个字典(注意, 这里重写了C2f模块和SPPF模块)"""
    pruned_yaml = {}
    with open(cfg, encoding='ascii', errors='ignore') as f:
        model_yamls = yaml.safe_load(f)  # model dict
    # # Define model
    nc = model.model.nc
    pruned_yaml["nc"] = nc
    pruned_yaml["scales"] = model_yamls["scales"]
    pruned_yaml["backbone"] = [
        [-1, 1, Conv, [64, 3, 2]],  # 0-P1/2
        [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
        [-1, 3, C2fpruned, [128, True]],
        [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
        [-1, 6, C2fpruned, [256, True]],
        [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
        [-1, 6, C2fpruned, [512, True]],
        [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
        [-1, 3, C2fpruned, [1024, True]],
        [-1, 1, SPPFpruned, [1024, 5]],  # 9
    ]
    pruned_yaml["head"] = [
        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        [[-1, 6], 1, Concat, [1]],  # cat backbone P4
        [-1, 3, C2fpruned, [512]],  # 12
        
        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        [[-1, 4], 1, Concat, [1]],  # cat backbone P3
        [-1, 3, C2fpruned, [256]],  # 15 (P3/8-small)
        
        [-1, 1, Conv, [256, 3, 2]],
        [[-1, 12], 1, Concat, [1]],  # cat head P4
        [-1, 3, C2fpruned, [512]],  # 18 (P4/16-medium)
        
        [-1, 1, Conv, [512, 3, 2]],
        [[-1, 9], 1, Concat, [1]],  # cat head P5
        [-1, 3, C2fpruned, [1024]],  # 21 (P5/32-large)
        
        [[15, 18, 21], 1, Detect, [nc]], # Detect(P3, P4, P5)
    ]
    # ========================================step7=========================================
    """
    遍历模型的每一个module, 按照剪枝阈值计算掩码, 然后将bn层对应的weight和bias乘以掩码; 
    计算每一层的原本的channel数, 剪枝后的channel数, 打印出来; 
    按照{bn层名字: bn层mask掩码矩阵}保存出一个maskbndict
    """
    print("=" * 94)
    print(f"|\t{'layer name':<25}{'|':<10}{'origin channels':<20}{'|':<10}{'remaining channels':<20}|")
    for name, module in model.model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            origin_channels = module.weight.data.size()[0]
            remaining_channels = origin_channels
            if name not in ignore_bn_list:
                mask = module.weight.data.abs().gt(thre).float()
                assert mask.sum() > 0, f"bn {name} has no active elements"
                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)
                remaining_channels = mask.sum().int()
            print(f"|\t{name:<25}{'|':<10}{origin_channels:<20}{'|':<10}{remaining_channels:<20}|")
    print("=" * 94)
                

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'ultralytics/cfg/datasets/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov8s.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default=ROOT / 'ultralytics/cfg/models/v8/yolov8.yaml', help='model.yaml path')
    parser.add_argument('--model-size', type=str, default='s', help='(yolov8)n, s, m, l or x?')
    parser.add_argument('--prune-ratio', type=float, default=0.2, help='prune ratio')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
