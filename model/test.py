import torchvision.models
import torch
from torchvision.models import resnet18
from torchsummary import summary
model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
summary(model=model, input_size=(3, 256, 256), batch_size=8, device='cpu')