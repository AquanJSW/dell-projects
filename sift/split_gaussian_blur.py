"""
Gaussian Blur using opencv and gpu accelerate
Note: Split Gaussian Kernel is used
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import argparse

path = './549.jpg'

parser = argparse.ArgumentParser(description='Split Gaussian Blur')
parser.add_argument('-p', '--path', default=path, type=str, help="image's path")
parser.add_argument('-k', '--kernel_size', default=3, type=int, help="kernel size, must be an odd")
parser.add_argument('-s', '--sigma', default=1.0, type=float, help="value of sigma")
parser.add_argument('-nc', '--no_cuda', default=False, action='store_const', const=True,
                    help="no CUDA once specified")
parse = parser.parse_args()

device = torch.device('cuda' if (not parse.no_cuda) and torch.cuda.is_available() else 'cpu')


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


def main(img=parse.path, kernel_size=parse.kernel_size, sigma=parse.sigma):
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
    main()

