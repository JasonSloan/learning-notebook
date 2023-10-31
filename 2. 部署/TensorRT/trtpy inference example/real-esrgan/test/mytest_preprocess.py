import torch
import torch.nn as nn
import numpy as np

class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Assuming x has the shape [H, W, C] and channel order BGR
        x = x[:, :, :, [2, 1, 0]]
        x = x.permute(0, 3, 1, 2)
        x = x / 255.0  # Normalize
        return x
    
if __name__ == '__main__':
    image = torch.randn(1, 64, 64, 3)
    model = Preprocess()
    model.eval()
    output = model(image)
    print(output.shape)
