import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv3d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)


class FSRCNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Conv3d(1, 56, kernel_size=5, padding=2)
        self.shrink = nn.Conv3d(56, 12, kernel_size=1)
        self.mapping = nn.Sequential(
            nn.Conv3d(12, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(12, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(12, 12, kernel_size=3, padding=1)
        )
        self.expand = nn.Conv3d(12, 56, kernel_size=1)
        self.deconv = nn.ConvTranspose3d(56, 1, kernel_size=9, padding=4)

    def forward(self, x):
        x = F.relu(self.feature(x))
        x = F.relu(self.shrink(x))
        x = self.mapping(x)
        x = F.relu(self.expand(x))
        return self.deconv(x)
