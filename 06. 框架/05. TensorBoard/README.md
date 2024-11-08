```python
# 安装:pip install tensorflow
from torch.utils.tensorboard import SummaryWriter
```

# 一、单条曲线

```python
writer = SummaryWriter(log_dir=summury_dir)
for epoch in range(epochs):
    ...
    writer.add_scalar(tag="train_loss",scalar_value=loss,global_step=epoch)
writer.close()
```

# 二、多条曲线

```python
writer = SummaryWriter(log_dir=summury_dir)
for epoch in range(epochs):
    ...
    writer.add_scalars(main_tag="train_test_loss",tag_scalar_dict={"train_loss":train_loss,"test_loss":test_loss},global_step=epoch)
writer.close()
```

# 三、图片

```python
writer = SummaryWriter(log_dir=summury_dir)
for epoch in range(epochs):
    ...
    for imgs, labels in data_loader:
        grid = torchvision.utils.make_grid(imgs,nrow=4)
        writer.add_image('images', grid, epoch)
        writer.add_graph(network, imgs)
writer.close()
```

# 四、将matplotlib中的图片加载到tensorboard中

```python
figure = plt.figure()
plt.plot(x, y, 'r-')
writer.add_figure('my_figure', figure, 0)
writer.close()
```

# 五、模型

```python
writer = SummaryWriter(log_dir=summury_dir)
for epoch in range(epochs):
    ...
    for imgs, labels in data_loader:
        writer.add_graph(network, imgs)      # 在tensorboard中双击可以展开模型
writer.close()
```

