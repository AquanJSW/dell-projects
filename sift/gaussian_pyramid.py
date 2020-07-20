"""Gaussian Pyramid
"""

import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from split_gaussian_blur import GaussianBlurNet
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser(description="Generate gaussian pyramid")
parser.add_argument('-p', '--path', default="./549.jpg", type=str, help="image's path")
parser.add_argument('-S', default=3, type=int, help="interval layers number = S + 3")
parser.add_argument('-s0', '--sigma0', default=1.6, type=float, help="the value of sigma_0")
parse = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GaussianPyramid:
    """主函数
    注意计算高斯金字塔组数时，考虑组内最后一张图像的高斯模板大小k，金字塔顶层的图像大小应 >= odd(k / 2)
    """
    def __init__(self, path=parse.path, S=parse.S, sigma0=parse.sigma0):
        self.img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.S = S  # 高斯金字塔的组内层数 = S + 3
        self.sigma0 = sigma0  # 初始高斯模糊的方差
        self.O = int(np.log2(self.img.shape).min() - np.log2(16)) + 1    # 高斯金字塔组数
        net = GaussianPyramidNet(img=self.img, S=self.S, sigma0=self.sigma0, O=self.O)
        self.pyramid = net()

    def get(self):
        """返回张量形式的高斯金字塔
        """
        return self.pyramid

    def show(self):
        pyramid = self._to_numpy()
        num = self.S + 3
        for i, img in enumerate(pyramid):
            plt.figure(i // num)
            plt.subplot(num, 1, i % num + 1), plt.imshow(img, 'gray')
        plt.show()

    def write(self):
        """输出为图像
        """
        t = time.ctime(time.time())
        pyramid = self._to_numpy()
        for i, img in enumerate(pyramid):
            o = i // self.O
            s = i % self.O
            cv2.imwrite('./output/o%ds%d-%s.jpg' % (o, s, t), img)

    def _to_numpy(self):
        pyramid_ = list()
        for img in self.pyramid:
            pyramid_.append(img.to(device='cpu', dtype=torch.uint8).squeeze().numpy())
        return pyramid_


class GaussianPyramidNet(nn.Module):
    """该网络直接生成高斯金字塔
    """
    def __init__(self, img, S, sigma0, O):
        super(GaussianPyramidNet, self).__init__()
        self.img = img
        self.S = S
        self.sigma0 = sigma0
        self.O = O
        self.one = torch.tensor([[[[1.]]]]).to(device=device, dtype=torch.float)

    def forward(self):
        pyramid = list()
        pyramid.append(preprocess(self.img))

        # 生成组内尺度坐标
        k = np.power(2, 1 / self.S)
        sigma = list()
        for s in range(1, self.S + 3):
            pre = self.sigma0 * (k ** (s - 1))
            now = pre * k
            sigma.append((now ** 2 - pre ** 2) ** 0.5)

        for o in range(self.O + 1):
            """组循环
            """
            if o == 0:
                pass
            else:
                """新组的第一张图像由上一组倒数第三张图像隔点下采样得到
                """
                pyramid.append(F.conv2d(input=pyramid[o * (self.S + 3) - 3],
                                        weight=self.one, stride=2))
            for s in range(1, self.S + 3):
                """层循环
                对上一张图像进行卷积
                """
                net = GaussianBlurNet_(compute_kernel(sigma[s - 1]))
                pyramid.append(net(pyramid[o * (self.S + 3) + s - 1]))

        return pyramid


class GaussianBlurNet_(GaussianBlurNet):
    """高斯模糊
    """
    def __init__(self, kernel):
        super(GaussianBlurNet_, self).__init__(kernel)

    def forward(self, img):
        """
        :param img: type is np or tensor
        :return: type is tensor
        """
        if not torch.is_tensor(img):
            img = torch.from_numpy(img).to(device=device, dtype=torch.float)
            img = img.unsqueeze(0).unsqueeze(0)
        """Reflect padding~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        """
        padH = nn.ReflectionPad2d((0, 0, self._padding, self._padding))
        padW = nn.ReflectionPad2d((self._padding, self._padding, 0, 0))
        img = padH(img)
        img = F.conv2d(img, weight=self._vkernel)
        img = padW(img)
        img = F.conv2d(img, weight=self._hkernel)
        return img


def compute_kernel(sigma):
    """计算高斯卷积核大小
    """
    kernel_size = int(sigma * 6 + 1)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size - 1
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    return kernel


def preprocess(img):
    """图像预处理
    1 高斯模糊抗混淆
    2 双线性插值放大一倍
    :param img: type is np
    :return: type is tensor
    """
    sigma = 0.5  # 抗击混淆所用的高斯模糊方差
    net = GaussianBlurNet_(compute_kernel(sigma))
    img = net(img)
    img = F.interpolate(img, scale_factor=2, mode='bilinear')
    return img
