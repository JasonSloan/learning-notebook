[网上总结的一些训练经验](https://www.zhihu.com/question/25097993/answer/3410497378?utm_campaign=shareopn&utm_medium=social&utm_psn=1781226666696081408&utm_source=wechat_session)

## 一. Batch Size

1. batch size增大了k倍，相应的学习率也应该增大k倍；
2. 在CV大多数数据集上，batch size设置为32最佳；
3. ​

## 二. Learning Rate

1. learing rate应使用warmup策略，即在开始的几个epochs内，learing rate从0开始线性增大到一定值，然后再执行正常的learning rate schedule计划。

## 三. Optimizer

1. 如果learning rate变化了，如果是使用带有动量（momentum）的优化器，那么momentum应该做校正，校正因子为 $$\frac{\eta_{t}+1}{\eta_{t}}$$ ，也就是新的momentum是原momentum的 $$\frac{\eta_{t}+1}{\eta_{t}}$$ 倍；

##四. Batch Normalization

1. 在初始化模型的时候，对于BN层的gamma和beta的初始化，一般是将所有的gamma初始化为1，beta初始化为0。但是如果将残差块中最后一个BN层的gamma初始化为0，beta初始化为0，其余位置的BN层的gamma依然初始化为1，beta初始化为0，会有效果提升；



## 二. 其他

### 1. 有用的操作:

迁移比从头训效果好; 

从在低分辨率的数据上训好的的模型迁移不如直接从官方模型迁移;

focal_loss有用;

使用更大的模型有用(在此实验上可以卡出能提高置信度阈值);

使用官方预训练模型先训练coco数据集后再训练自己的数据集有用, 涨点不少;

增加自己的训练数据集大小有用, 涨点很多;

优先使用官方的超参数, 超参并不能迁移(在数据集1上效果最好的超参放在数据集2上并不一定work)

### 2. 无用的操作:

从低分辨率的最优超参数训出的模型不如官方最优超参数训出的;

冻结参数过多;

### 3. 是否有用视情况而定的操作:

增加训练轮数

增大输入图像的分辨率

多尺度

更大的数据增强