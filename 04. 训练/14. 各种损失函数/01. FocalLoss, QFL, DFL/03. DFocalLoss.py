import torch
import torch.nn as nn
import torch.nn.functional as F

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


def dfl_loss(pred_dist, target):
    """Return sum of left and right DFL losses.

    Args:
        pred_dist (Tensor): shape: [?*4, 16], 4代表边框的4个边, ?代表有物体的单元格的个数, 16代表着16个位置的概率
        target (Tensor): shape: [?, 4], ?代表有物体的单元格的个数, 4代表着边框的4个边

    Returns:
        loss: shape: [?, 1], ?代表有物体的单元格的个数
    """
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    tl = target.long()  # target left, 真实框距离单元格中心点位置的左侧(上侧)索引
    tr = tl + 1         # target right, 真实框距离单元格中心点位置的右侧(下侧)索引
    wl = tr - target    # weight left, 左侧权重
    wr = 1 - wl         # weight right, 右侧权重
    # 多标签交叉熵损失, 使16个预测概率的第tl和tr两个位置的概率最大
    return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
            F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)
    

if __name__ == '__main__':
    # 假如batch-size为1; 输出的最后三张特征图为80*80,40*40,20*20; 64代表框的4个边, 每个边预测16个位置的概率
    last_three_feature_maps = [
        torch.randn(1, 64, 80, 80, dtype=torch.float32),
        torch.randn(1, 64, 40, 40, dtype=torch.float32),
        torch.randn(1, 64, 20, 20, dtype=torch.float32)
    ]
    last_three_feature_maps = [xi.view(1, 64, -1) for xi in last_three_feature_maps]        
    last_three_feature_maps = torch.cat(last_three_feature_maps, dim=2)                     # [1, 64, 8400]
    pred_dist = last_three_feature_maps.view(1, -1, 4, 16).softmax(3)                       # [1, 8400, 4, 16]
    # # 如果是推理阶段, 就使用这个DFL类把16个距离转换成1个值, DFL类没有参数, 这么写可以方便的将这个后处理加入到整个网络中
    # dfl = DFL()
    # pred_boxes = dfl(last_three_feature_maps)
    # print(pred_boxes.shape)
    # 假设第50个单元格是有真实框的, 其他的单元格都是背景
    fg_mask = torch.zeros([1, 8400]).type(torch.bool)                       
    fg_mask[0, 50] = 1
    # 在有真实框的单元格位置上预测的4个边距离单元格中心的16个位置的概率
    pred_dist = pred_dist[fg_mask].view([-1, 16])
    # 真实框的四个边距离所在单元格中心的距离
    target = torch.tensor([[2.3, 12.1, 5.4, 8.5]])
    loss = dfl_loss(pred_dist, target)
    print(loss.item())