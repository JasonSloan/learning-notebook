# 1. 使用setattr将模型中的指定模块替换为其他模块

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

# 2. 如果有这么个需求，有一个类，他的成员方法中使用到了类的成员属性，在不改变原代码的情况下，如何替换掉这个方法？

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

