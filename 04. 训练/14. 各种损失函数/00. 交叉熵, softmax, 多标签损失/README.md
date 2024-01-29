## 多标签损失

$\text{BCE Loss}(\mathbf{y}, \mathbf{\hat{y}}) = -\sum_{i=1}^{K} \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right)$

```python
import torch
import torch.nn as nn

def multi_label_loss_bce(preds, targets):
    """非内置sigmoid的多标签损失函数

    Args:
        preds (_type_): _description_
        targets (_type_): _description_

    Returns:
        _type_: _description_
    """
    criterion = nn.BCELoss(reduction="sum")
    loss = criterion(preds.sigmoid(), targets)
    return loss

def multi_label_loss_bce_with_logits(preds, targets):
    """内置sigmoid的多标签损失函数, 官方推荐

    Args:
        preds (_type_): _description_
        targets (_type_): _description_

    Returns:
        _type_: _description_
    """
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    loss = criterion(preds, targets)
    return loss

def multi_label_loss_custom(preds, targets):
    """自己实现的多标签损失函数

    Args:
        preds (_type_): _description_
        targets (_type_): _description_

    Returns:
        _type_: _description_
    """
    batch_size, classes = targets.shape
    loss = 0
    for i in range(batch_size):
        for j in range(classes):
            preds_sigmoid = preds[i][j].sigmoid()
            if targets[i][j] == 1:
                loss += torch.log(preds_sigmoid)
            else:
                loss += torch.log(1 - preds_sigmoid)
    return -loss  


if __name__ == "__main__":
    targets = torch.tensor([[1, 0, 1],
                            [0, 1, 1]], dtype=torch.float32)
    preds = torch.tensor([[0.5982, -0.7341, 0.2625],
                        [0.3148,  1.1125, -0.0490]], dtype=torch.float32)

    loss_bce = multi_label_loss_bce(preds, targets)
    print("BCELoss:", loss_bce.item())

    loss_bce_with_logits = multi_label_loss_bce_with_logits(preds, targets)
    print("BCEWithLogitsLoss:", loss_bce_with_logits.item())

    loss_custom = multi_label_loss_custom(preds, targets)
    print("Custom Multi-Label Loss:", loss_custom.item())
```

