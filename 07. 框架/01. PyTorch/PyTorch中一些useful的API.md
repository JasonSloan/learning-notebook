### 1. tensor.masked_fill_

对于一个tensor， 按照给定mask掩膜， 对掩膜中bool值为True的地方填充value

```python
tensor = torch.randn(2, 3)
print(tensor)
mask = torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.bool)
print(mask)
tensor.masked_fill_(mask, 0)
print(tensor)
>> tensor([[-0.6768,  0.3607,  0.6569],
           [ 0.1292,  1.8583, -0.3451]])
>> tensor([[False,  True, False],
           [False, False,  True]])
>> tensor([[-0.6768,  0.0000,  0.6569],
           [ 0.1292,  1.8583,  0.0000]])
```

### 2. torch.gather

对于一个tensor, 指定某一维度, 按照指定索引获得数据并聚集到一起

```python
>>> tensor = torch.arange(12).reshape(3,4)
>>> tensor
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> idx = torch.tensor([[0, 2, 2]])	# 将第1行索引为0, 第2行索引为2, 第3行索引为2的元素gather到一起
>>> torch.gather(tensor, 0, idx)	# dim指定为0(指定为1会出现奇怪的结果)
tensor([[ 0, 6, 10]])				
```

### 3. 使用nn.Embedding代替nn.Parameter

在构建模型时, 有的时候有些tensor变量需要再init中声明, 然后在forward中使用, 而且该变量需要可以梯度更新, 常用做法是使用nn.Parameter, 但是使用nn.Embedding会更方便

```python
nn.Parameter(torch.rand(2,3), requires_grad=True)
Parameter containing:
tensor([[0.5926, 0.3559, 0.1848],
        [0.6024, 0.4757, 0.2214]], requires_grad=True)
nn.Embedding(2,3).weight
Parameter containing:
tensor([[ 0.0274,  1.2454,  0.2842],
        [-0.4236,  0.5645, -1.9724]], requires_grad=True)


# 下面的例子使用nn.Embedding代替nn.Parameter
class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.embedding = nn.Parameter(torch.randn(1, 3, 1, 1), requires_grad=True)
        self.embedding = nn.Embedding(1, 3).weight.unsqueeze(-1).unsqueeze(-1)
        self.conv = nn.Conv2d(3, 128, 3, 1, 1)
        self.bn = nn.BatchNorm2d(128)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x += self.embedding
        return self.act(self.bn(self.conv(x)))

if __name__ == "__main__":
    dummy = torch.randn(1, 3, 640, 640)
    model = Model()
    model(dummy)

```

### 4. F.interpolate

对于一个tensor（至少为3维）, 在最后一个（或最后两个）维度进行插值（取决于是单线性插值还是双线性插值）

```python
>>> t = torch.arange(0,24).reshape(2,3,4).float()
>>> t
tensor([[[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.]],

        [[12., 13., 14., 15.],
         [16., 17., 18., 19.],
         [20., 21., 22., 23.]]])
>>> F.interpolate(t, size=5, mode="linear")
tensor([[[ 0.0000,  0.7000,  1.5000,  2.3000,  3.0000],
         [ 4.0000,  4.7000,  5.5000,  6.3000,  7.0000],
         [ 8.0000,  8.7000,  9.5000, 10.3000, 11.0000]],

        [[12.0000, 12.7000, 13.5000, 14.3000, 15.0000],
         [16.0000, 16.7000, 17.5000, 18.3000, 19.0000],
         [20.0000, 20.7000, 21.5000, 22.3000, 23.0000]]])
>>> t = t[None]		# 扩大到4维（shape:[1,2,3,4]）
>>> F.interpolate(t, size=5, mode="bilinear")
tensor([[[[ 0.0000,  0.7000,  1.5000,  2.3000,  3.0000],
          [ 1.6000,  2.3000,  3.1000,  3.9000,  4.6000],
          [ 4.0000,  4.7000,  5.5000,  6.3000,  7.0000],
          [ 6.4000,  7.1000,  7.9000,  8.7000,  9.4000],
          [ 8.0000,  8.7000,  9.5000, 10.3000, 11.0000]],

         [[12.0000, 12.7000, 13.5000, 14.3000, 15.0000],
          [13.6000, 14.3000, 15.1000, 15.9000, 16.6000],
          [16.0000, 16.7000, 17.5000, 18.3000, 19.0000],
          [18.4000, 19.1000, 19.9000, 20.7000, 21.4000],
          [20.0000, 20.7000, 21.5000, 22.3000, 23.0000]]]])

```



