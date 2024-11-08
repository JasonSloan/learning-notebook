# pip install torchinfo
from torchinfo import summary
from torchvision.models import resnet18

if __name__ == "__main__":
    model = resnet18()
    summary(model, (1, 3, 224, 224))