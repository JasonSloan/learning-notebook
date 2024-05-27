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

### 3. 使用nn.Embedding代替nn.Parameter

