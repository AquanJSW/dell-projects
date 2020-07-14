"""
Gaussian Blur using opencv and gpu accelerate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device("cuda")
path0 = 'd:\\Dataset\\steampunk.png'
path1 = 'd:\\Dataset\\msl\\left_nav_pair\\10286.jpg'
path2 = '/home/tjh/dataset/steampunk.png'


# 用于高斯模糊
class GaussianBlurNet(nn.Module):
    def __init__(self, kernel):
        super(GaussianBlurNet, self).__init__()
        self.padding = len(kernel) // 2
        kernel = torch.from_numpy(kernel).to(device=device, dtype=torch.float16)
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)
        # self.kernel = nn.Parameter(self.kernel, requires_grad=False)

    def forward(self, img):
        img_ = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float16)
        img_ = F.conv2d(img_, weight=self.kernel, padding=self.padding)
        img_ = img_.squeeze().to(device='cpu', dtype=torch.uint8).numpy()
        return img_


def main(img="/home/tjh/dataset/msl/left_nav_pair/10268.jpg",
         kernel_size=7, sigma=18):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # cv2.namedWindow('before process', cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow('before process', img)
    # k = cv2.waitKey()
    # if k == 's':
    #     cv2.destroyWindow('before process')
    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('before process')
    kernel = cv2.getGaussianKernel(kernel_size, sigma)  # 返回ndarray, 1D gaussian kernel
    kernel = np.dot(kernel, kernel.T)   # 2D gaussian kernel
    print(kernel)
    print(kernel.sum())
    net = GaussianBlurNet(kernel)
    start = time.time()
    img_ = net(img)
    stop = time.time()
    print(stop - start)
    # img_ = img_ * 255 / img_.max()
    # img_ = np.asarray(img_, dtype=np.int)
    plt.subplot(122), plt.imshow(img_, 'gray'), plt.title('after process')
    plt.show()
    print(img_.shape)


if __name__ == '__main__':
    main(img=path1)

