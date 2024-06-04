import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18
import matplotlib.pyplot as plt


if __name__ == "__main__":
    epochs = 100
    init_lr = 0.1
    min_lr = 0.001
    resnet = resnet18()
    optimizer = Adam(resnet.parameters(), lr=init_lr)
    lr_scheduler = CosineAnnealingLR(optimizer, 32, eta_min=min_lr)
    lrs = []
    for i in range(epochs):
        current_lr = lr_scheduler.get_lr()
        lr_scheduler.step()
        lrs.append(current_lr)
    plt.figure()
    plt.plot(list(range(epochs)), lrs, marker='o')
    plt.title('CosineAnnealingLR')
    plt.xlabel('epochs')
    plt.ylabel('lr')
    plt.legend()
    plt.savefig("cosine.jpg")
