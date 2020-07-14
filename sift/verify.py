import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.kernel = torch.tensor([0.1, 0.1, 0.1]).reshape(1, 3).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float16)

    def forward(self, x):
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float16)
        x = F.conv2d(x, weight=self.kernel, padding=(1, 0))
        return x


if __name__ == '__main__':
    net = Net()
    x = np.array([255, 200, 30, 4]).reshape((2, 2))
    x = net(x)