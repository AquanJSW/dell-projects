# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #x = F.max_pool2d(x, 2) # 两层池化使128图变为32图
        #x = F.max_pool2d(x, 2)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x: batch * channel * h * w
        # num_flat_features = channel * h * w
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# nn.Module.parameters() 是一个生成器，包含所有参数
# 看下面的输出，多维的size是weights，它与网络结构相同
# 常量size是bias，等于对应层的输出channel数
learnable_params = list(net.parameters())
print(len(learnable_params))
for i in learnable_params:
    print(i.shape)

I = Image.open('./img.jpeg')
I = np.array(I)
# 下面两行效果一样
# I = torch.from_numpy(I).resize_(1, 1, 128, 128)
I = torch.from_numpy(I).unsqueeze_(0).unsqueeze_(0)
print(I.shape)

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
# net.zero_grad()
# out.backward(torch.randn(1, 10))

# 定义loss
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)

# net.zero_grad()
# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)
# loss.backward()
# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)

# net.zero_grad()
# # 手动梯度下降法
# learning_rate = 0.01
# c = 0
# loss.backward()
# while c < 5:
#     c += c
#     for p in net.parameters():
#         p.data.sub_(learning_rate * p.grad.data)
#     print(loss)

optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
loss.backward()
optimizer.step()
