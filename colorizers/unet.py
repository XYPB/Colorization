import torch
import torch.nn as nn
import numpy as np
from torch.nn import init

class tinyUnet(nn.Module):
    def __init__(self):
        super(tinyUnet, self).__init__()
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
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.usample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.unit3 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.unit4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1),
        )

    def forward(self, x):
        x = self.unit1(x)
        u1 = x
        x = self.dsample(x) # 32
        x = self.unit2(x)
        u2 = x
        x = self.dsample(x) # 16
        x = self.usample(self.unitM(x)) # 32
        x = torch.cat([x, u2],1)
        x = self.usample(self.unit3(x)) # 64
        x = torch.cat([x, u1],1)
        x = self.unit4(x)
        return x

def unet(pretrained=False, model_path=None):
    model = tinyUnet()
    if pretrained:
        model.load_state_dict(torch.load(model_path))
    return model