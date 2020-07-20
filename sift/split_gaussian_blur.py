"""
Gaussian Blur using openCV and gpu acceleration
Note: Split Gaussian Kernel is used

API version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GaussianBlurNet(nn.Module):
    def __init__(self, kernel):
        """
        :param kernel: from cv2.getGaussianKernel()
        """
        super(GaussianBlurNet, self).__init__()
        self.train(False)
        # 两个一维卷积核
        # 可以不使用nn.Parameter，因为该方法将v/hkernel转化为可训练的参数，但这里根本不用训练
        vkernel = torch.from_numpy(kernel).to(device=device, dtype=torch.float)
        hkernel = vkernel.T
        self._vkernel = vkernel.unsqueeze(0).unsqueeze(0)
        self._hkernel = hkernel.unsqueeze(0).unsqueeze(0)
        self._padding = len(kernel) // 2

    # 两次卷积实现分离高斯模糊
    def forward(self, img):
        img = torch.from_numpy(img).to(device=device, dtype=torch.float)
        img = img.unsqueeze(0).unsqueeze(0)
        img = F.conv2d(img, weight=self._vkernel, padding=(self._padding, 0))
        img = F.conv2d(img, weight=self._hkernel, padding=(0, self._padding))
        img = img.squeeze().to(device='cpu', dtype=torch.uint8).numpy()
        return img


class GaussianBlur:
    """Use this class
    """
    def __init__(self, img, kernel_size, sigma):
        # 核心非奇数返回Error
        if not (kernel_size % 2):
            raise ValueError("kernel size must be and odd")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.img = img
        self.kernel = cv2.getGaussianKernel(kernel_size, sigma)
        net = GaussianBlurNet(self.kernel)
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
