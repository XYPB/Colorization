import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from IPython import embed

class tinyUnet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.unit1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.dsample = nn.MaxPool2d(2, stride=2)
        self.unit2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.unitM = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.usample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.unit3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.unit4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 5, 3, padding=1),
        )

    def forward(self, x):
        x = self.unit1(x)
        u1 = x
        x = self.dsample(x)
        x = self.unit2(x)
        u2 = x
        x = self.dsample(x)
        x = self.usample(self.unitM(x))
        x = x + u2
        x = self.usample(self.unit3(x))
        x = x + u1
        x = self.unit4(x)
        return x

def unet(pretrained=False, model_path=None):
    model = tinyUnet()
    if pretrained:
        model.load_state_dict(torch.load(model_path))
    return model