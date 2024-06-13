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

