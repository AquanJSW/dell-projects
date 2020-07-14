"""
Gaussian Blur using opencv and gpu accelerate
Note: Split Gaussian Kernel is used
!!! 用pytorch实现简单地一次卷积，结果可能并非正常!!!
        用简单地卷积验证一下
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda")
path1 = 'd:\\Dataset\\msl\\left_nav_pair\\10286.jpg'


# 用于高斯模糊
class GaussianBlurNet(nn.Module):
    def __init__(self, kernel):
        super(GaussianBlurNet, self).__init__()
        self.train(False)
        # 两个一维卷积核
        # 可以不使用nn.Parameter，因为该方法将v/hkernel转化为可训练的参数，但这里根本不用训练
        vkernel = torch.from_numpy(kernel).to(device=device, dtype=torch.float16)
        hkernel = vkernel.T
        self.vkernel = vkernel.unsqueeze(0).unsqueeze(0)
        self.hkernel = hkernel.unsqueeze(0).unsqueeze(0)
        self.padding = len(kernel) // 2

    def forward(self, img):
        img = torch.from_numpy(img).to(device=device, dtype=torch.float16)
        img = img.unsqueeze(0).unsqueeze(0)
        img = F.conv2d(img, weight=self.vkernel, padding=(0, self.padding))
        img = F.conv2d(img, weight=self.hkernel, padding=(self.padding, 0))
        img = img.squeeze().to(device='cpu', dtype=torch.uint8).numpy()
        return img


def gaussian_blur(img="/home/tjh/dataset/msl/left_nav_pair/10268.jpg",
         kernel_size=7, sigma=18):
    # 核心非奇数返回Error
    if not (kernel_size % 2):
        raise ValueError("kernel size must be odd")

    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)                      # 读取并转换为灰度图
    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('origin')   # 绘制处理前的图像
    kernel = cv2.getGaussianKernel(kernel_size, sigma)               # 得到一维Gaussian核，ndarray type
    net = GaussianBlurNet(kernel)
    img = net(img)                                                   # 返回Gaussian模糊后的图像
    plt.subplot(122), plt.imshow(img, 'gray'),\
        plt.title('kernel: %d, sigma: %.1f' % (kernel_size, sigma))  # 绘制处理后的图像
    plt.show()


if __name__ == '__main__':
    gaussian_blur()

