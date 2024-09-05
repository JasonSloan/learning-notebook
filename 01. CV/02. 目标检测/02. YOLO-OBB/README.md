## 一. 旋转框之间的IoU计算方法

[原论文](https://arxiv.org/pdf/2106.06072v1.pdf)

思想:

将HBB(Horizontal Bounding Box)或者OBB(Oriented Bounding Box)建模为GBB(Gaussian Bouding Box) , 从而得到框在x和y方向的均值和协方差;

利用GBB的均值和协方差计算每两个框之间的[巴氏距离](https://blog.csdn.net/hy592070616/article/details/122400635?ops_request_misc=%257B%2522request%255Fid%2522%253A%25220259A412-4933-4D44-A22D-2363B013C43C%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=0259A412-4933-4D44-A22D-2363B013C43C&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-122400635-null-null.142^v100^pc_search_result_base4&utm_term=bhattacharyya%20distance&spm=1018.2226.3001.4187), 再通过巴氏距离计算[海林格距离](https://blog.csdn.net/hy592070616/article/details/122400635?ops_request_misc=%257B%2522request%255Fid%2522%253A%25220259A412-4933-4D44-A22D-2363B013C43C%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=0259A412-4933-4D44-A22D-2363B013C43C&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-122400635-null-null.142^v100^pc_search_result_base4&utm_term=bhattacharyya%20distance&spm=1018.2226.3001.4187)从而得到最终的IoU。

Note: [多维高斯分布的表示方法参考链接](https://blog.csdn.net/weixin_38468077/article/details/103508072?ops_request_misc=&request_id=&biz_id=102&utm_term=2d%E9%AB%98%E6%96%AF&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-103508072.142^v100^pc_search_result_base4&spm=1018.2226.3001.4187)

### 1. 将HBB转换为GBB

将一个xywh的HBB建模为一个均值为 $$\mu$$ 方差为 $$\sum$$ 的GBB, 如何通过xywh求 $$\mu$$ 和 $$\sum$$ 的方法:

 $\mu = \frac{1}{N} \int_{x \in \Omega} x$
 $\sum =\frac{1}{N}\int_{x\in\Omega} (x-\mu)^T(x-\mu)$

其中 $$\Omega$$ 代表HBB的区域, x是属于 $$\Omega$$ 内的任意一点, N代表共有多少个点

那么对于一个HBB矩形框来说, 以矩形框的中心点为原点(也就是 $$\mu$$ 的值为[0, 0])计算得到的 $$\sum$$ 为:

![](assets/1.jpg)

我们可以令![](assets/2.jpg)

### 2. 将一个OBB转换为GBB

OBB在HBB的基础上需要增加一个旋转角, 它的协方差矩阵  $$\sum$$ 对应的公式如下:

![](assets/3.jpg)

其中 $$a' = {W^2} / {12}$$ , $$b' = {H^2} / {12}$$

### 3. 计算两个GBB之间的巴氏距离

计算公式如下:

$$B_D=\frac{1}{8}(\mu_1-\mu_2)^T\Sigma^{-1}(\mu_1-\mu_2)+\frac{1}{2}\ln\left(\frac{\det\Sigma}{\sqrt{\det\Sigma_1\det\Sigma_2}}\right),\Sigma=\frac{1}{2}(\Sigma_1+\Sigma_2)$$

其中det代表求行列式的意思

而上述公式可以简化为:

$B_D=B_1+B_2$ 

其中:

![](assets/4.jpg)

### 4. 计算两个GBB之间的海林格距离

巴氏距离到海林格距离的转换公式:

![](assets/5.jpg)
![](assets/6.jpg)
最终将1 - hd的结果作为IoU的值

### 5. YOLOv8-OBB中的probiou的计算代码

```python
def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)     # x1,y1: [N, 1]
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))  # x2,y2: [1, N]
    a1, b1, c1 = _get_covariance_matrix(obb1)   # a1, b1, c1: [N, 1]
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))    # a2, b2, c2: [1, N]

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25    # t1: [N, N];   t1 + t2为公式(8)
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5 # t2: [N, N]    t1 + t2为公式(8)
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5   # t3: [N, N]    t3为公式(9)
    bd = (t1 + t2 + t3).clamp(eps, 100.0)   # bd: [N, N]    
    hd = (1.0 - (-bd).exp() + eps).sqrt()   # hd: [N, N]
    return 1 - hd


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance metrixs corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.(论文中的公式15)
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1) # 将OBB转换成GBB(guassion bounding  box):[N, 3]
    a, b, c = gbbs.split(1, dim=-1) # a: W^2/12;  b: H^2/12;  c: theta
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin    # 对应于公式(1)
```

