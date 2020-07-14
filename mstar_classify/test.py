# -*- coding: utf-8 -*-
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from gen_dataset import MyDataset
import numpy as np
from main import Net
from mydataloader import mydataloader

_, testloader = mydataloader()


def test(testloader=testloader, batch_size=16):

    classes = ('2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70',
               'D7', 'T62', 'T72', 'ZIL131', 'ZSU234')

    #device = torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    PATH = '/home/tjh/projects/py/mstar_classify/97mstar_trained_model.tar'
    net = Net().to(device=device)
    net.load_state_dict(torch.load(PATH))



    # Global correction
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device=device), data[1].to(device=device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (labels == predicted).sum().item()

    print("Accuracy of the network on the 2600 test images: %d %%" %
          (100 * correct / total))


    # Class-wise accuracy

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device=device), data[1].to(device=device)
            # 跳过最后不完整的batch
            if len(labels) - batch_size:
                continue
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (labels == predicted).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label.item()] += c[i].item()
                class_total[label.item()] += 1

    for i in range(10):
        print('Accuracy of %6s : %2d %%' %
              (classes[i], class_correct[i] * 100 / class_total[i]))


if __name__ == '__main__':
    test()
