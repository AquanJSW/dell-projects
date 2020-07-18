"""
Gaussian Blur using openCV and gpu acceleration
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
class _GaussianBlurNet(nn.Module):
    """Do not use this class
    """
    def __init__(self, kernel):
        super(_GaussianBlurNet, self).__init__()
        self.train(False)
        # 两个一维卷积核
        # 可以不使用nn.Parameter，因为该方法将v/hkernel转化为可训练的参数，但这里根本不用训练
        vkernel = torch.from_numpy(kernel).to(device=device, dtype=torch.float16)
        hkernel = vkernel.T
        self._vkernel = vkernel.unsqueeze(0).unsqueeze(0)
        self._hkernel = hkernel.unsqueeze(0).unsqueeze(0)
        self._padding = len(kernel) // 2

    # 两次卷积实现分离高斯模糊
    def forward(self, img):
        img = torch.from_numpy(img).to(device=device, dtype=torch.float16)
        img = img.unsqueeze(0).unsqueeze(0)
        img = F.conv2d(img, weight=self._vkernel, padding=(0, self._padding))
        img = F.conv2d(img, weight=self._hkernel, padding=(self._padding, 0))
        img = img.squeeze().to(device='cpu', dtype=torch.uint8).numpy()
        return img


class GaussianBlur:
    """Use this class
    """
    def __init__(self, path=parse.path, kernel_size=parse.kernel_size, sigma=parse.sigma):
        # 核心非奇数返回Error
        if not (kernel_size % 2):
            raise ValueError("kernel size must be and odd")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.kernel = cv2.getGaussianKernel(kernel_size, sigma)
        net = _GaussianBlurNet(self.kernel)
        self.out = net(self.img)

    def show(self):
        """Show origin and output image
        """
        plt.subplot(121), plt.imshow(self.img, 'gray'), plt.title('origin')
        plt.subplot(122), plt.imshow(self.out, 'gray')
        plt.title('kernel: %d   sigma: %.1f' % (self.kernel_size, self.sigma))
        plt.show()

    def output(self):
        """Return numpy format output image
        """
        return self.out
