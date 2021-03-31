import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from skimage import color
import os

_transformer = transforms.Compose([
        transforms.Resize([128,128]),
        transforms.RandomHorizontalFlip(),
    ])

_mean, _std = torch.Tensor([0.485, 0.456, 0.406]), torch.Tensor([0.229, 0.224, 0.225])

class TinyCOCO(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        img, _ = super(TinyCOCO, self).__getitem__(index)
        img = torch.Tensor(color.rgb2lab(img))
        # img = (img - _mean) / _std
        return img[None,...,0], img[...,1:].permute(2,0,1)

def get_TinyCOCO_loader(batch_size=32,root='./dataset/COCO/', task='train', transfomer=_transformer):
    return DataLoader(TinyCOCO(os.path.join(root, task), transform=transfomer), shuffle=True, batch_size=batch_size)


