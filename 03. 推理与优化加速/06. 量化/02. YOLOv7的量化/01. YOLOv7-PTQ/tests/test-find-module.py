import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer1 = Conv(3, 32, 3, 1)
        self.layer2 = Conv(32, 32, 3, 1)
    
    def forward(self, x):
        return self.layer2(self.layer1(x))


def replace_basic_module(module):
    # 这里为什么不遍历named_module是因为Model模型中嵌套着Conv模块，在遍历named_module后无法按照name去替换nn.Conv2d
    for name in module._modules:
        submodule = module._modules[name]
        replace_basic_module(submodule)
        if isinstance(submodule, nn.Conv2d):
            module._modules[name] = nn.Linear(128, 256)


if __name__ == "__main__":
    model = Model()
    replace_basic_module(model)
    print(model)