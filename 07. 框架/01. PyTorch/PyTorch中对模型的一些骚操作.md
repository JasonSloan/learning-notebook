## 1. 使用setattr将模型中的指定模块替换为其他模块

```python
import torch
import torch.nn as nn

class Conv2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        conv_list = nn.ModuleList()
        for i in range(2):
            conv_list.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.conv2 = nn.Sequential(*conv_list)
    
    def forward(self, x):
        return self.conv2(x)
    
    
class Conv3(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        conv_list = nn.ModuleList()
        for i in range(3):
            conv_list.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.conv3 = nn.Sequential(*conv_list)
    
    def forward(self, x):
        return self.conv3(x)
    
    
class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.convfirst = nn.Conv2d(64, 64, 3, 1, 1)
        self.convsecond = Conv2()
        self.convthird = nn.Conv2d(64, 64, 3, 1, 1)
        
    def forward(self, x):
        return self.convthird(self.convsecond(self.convfirst(x)))


if __name__ == "__main__":
    model = Model()
    print(model)
    for name, module in model.named_modules():
        if isinstance(module, Conv2):
            conv3 = Conv3()
            # 使用setattr将conv2模块替换成conv3
            setattr(model, name, conv3)
    print("=" * 100)
    print(model)
```

## 2. 遍历module._modules将模型中的指定模块替换为其他模块

```python
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
```

## 3. 如果有这么个需求，有一个类，他的成员方法中使用到了类的成员属性，在不改变原代码的情况下，如何替换掉这个方法？

如下：

```python
class Trainer:
    def __init__(self, model="resnet18"):
        self.model = model

    def train(self):
        print(f"Training v1 wiht model {self.model}")


def train_v2(self: Trainer, extra_arg=False):
    print(f"Training v2 wiht model {self.model}")
    if extra_arg:
        print("Do sth with extra_arg")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    # 由于trainner.train方法中使用到了self.model, self.model是Trainer类的属性
    # 重写trainer类的train方法，也需要使用到trainer实例的self.model属性
    # 如果想使用trainer.train = train_v2来替换trainer的train方法是不可行的，因为这样无法在train_v2中使用到self.model了
    # 所以不改变任何原代码的情况下，需要使用trainer.__setattr__方法来设置trainer实例的属性
    trainer.__setattr__("train", train_v2.__get__(trainer))
    trainer.train()
```
## 4. 使用register_forward_hook可视化一个网络中的每一层的输出 

```python
import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = Conv(3, 32)
        self.conv2 = Conv(32, 32)
        self.conv3 = Conv(32, 64)

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))


def make_layer_forward_hook():
    def forward_hook(m, input, output):
        # visualize code 
        print(f'Module {m.__class__.__name__} output shape: {output.shape}')
    return forward_hook


if __name__ == "__main__":
    model = Model()
    for _, module in model.named_modules():
        if isinstance(module, Conv):
            module.register_forward_hook(make_layer_forward_hook())
    dummy = torch.randn(1, 3, 640, 640)
    model(dummy)
```

