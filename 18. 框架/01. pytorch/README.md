# 一. 一些不同于cuda的使用命令

```python
import torch
import torch_npu  #1.8.1及以上需要

# torch.cuda.is_available()
torch.npu.is_availabel()

# device = torch.device('cuda:0')
# torch.cuda.set_device(device)
device = torch.device('npu:0')
torch.npu.set_device(device)

# torch.cuda.device_count()
torch.npu.device_count()

# tensor.cuda()
tensor.npu()

# tensor.to("cuda:0")
tensor.to("npu:0")





```

