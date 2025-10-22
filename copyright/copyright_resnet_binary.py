import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet50Binary(nn.Module):
    def __init__(self):
        super(ResNet50Binary, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x