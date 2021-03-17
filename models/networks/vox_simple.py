import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVoxNet(nn.Module):
    def __init__(self, input, output):
        super(SimpleVoxNet, self).__init__()
        features = [64, 64, 32, 32]
        self.block0 = nn.Sequential(
            nn.Conv3d(input, features[0], kernel_size=3,padding=1),
            nn.BatchNorm3d(features[0]),
            nn.ReLU(True))
        self.block1 = nn.Sequential(
            nn.Conv3d(features[0], features[1], kernel_size=3,padding=1),
            nn.BatchNorm3d(features[1]),
            nn.ReLU(True))
        self.block2 = nn.Sequential(
            nn.Conv3d(features[1], features[2], kernel_size=3,padding=1),
            nn.BatchNorm3d(features[2]),
            nn.ReLU(True))
        self.block3 = nn.Sequential(
            nn.Conv3d(features[2], features[3], kernel_size=3,padding=1),
            nn.BatchNorm3d(features[3]),
            nn.ReLU(True))
        self.block4 = nn.Sequential(
            nn.Conv3d(features[3], output, kernel_size=3,padding=1))

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        return self.block4(self.block3(self.block2(self.block1(self.block0(x)))))
