import torch
import torch.nn as nn

class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


if __name__ == "__main__":
    loss_fcn = nn.BCEWithLogitsLoss()
    qfl = QFocalLoss(loss_fcn)
    # batch-size:2 classes:3
    pred = torch.tensor([[-1.7516,  0.4123,  0.0475],
                         [ 0.5953,  0.3117, -0.7500]], dtype=torch.float32)
    # multi-label
    true = torch.tensor([[0, 1, 0],
                         [1, 1, 0]], dtype=torch.float32)
    loss = qfl(pred, true)
    print(f"loss: {loss.item()}")