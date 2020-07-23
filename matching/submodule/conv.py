import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, block_size, device):
        super(Conv, self).__init__()
        self.block_size = block_size
        self.device = device
        self.padding = self.block_size // 2
        self.weight = torch.ones((self.block_size, self.block_size),
                                 dtype=torch.float, device=self.device)
        self.weight = self.weight.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        x = F.conv2d(x, self.weight)
        return x