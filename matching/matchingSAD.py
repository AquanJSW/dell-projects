"""SAD Matching for non-calibrated image pairs

IDEA before coding:
    class API with variety methods, including:
        device choice(using torch, anyway);
        parallel switch(to cpu and gpu, so totally 4 choices, but first achieve non-parallel with cpu)
        command-line saving and showing support(argparse: --show, --write)
        GET support (for other module)
        down-sampling options and gaussian blur options (both for command-line convenience)

    MODULE LIKE CODING !!!!!!
        for better debug

TODO:
    1 MAX DISPARITY may has a linear variation, try to take this in consideration
    2 using "edge detection" or "gaussian blur" to estimate MAX DISPARITY ??? then to fit the linear MAX DISPARITY line
        which is a parallel task(gpu accelerate)
    3 adaptable block size(considering the texture complexity)
    4 parameter VERTICAL SHIFT may be estimated through feature extraction??
    5 various penalties
    6 just like PSMNet's idea, for parallel computing, first compute a number of (MAX_DISPARITY * (SHIFT*2+1))
        cost maps(based on SAD), then for each pixel in left, log the minimal cost's (DISPARITY, SHIFT) coordination.
        finally, generate a tensor shape of [1,1,H,W,2], the last dim's content is (DISPARITY, SHIFT)
"""

import tqdm
import subprocess
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import shlex
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description="SAD Matching for non-calibrated image pairs")
parser.add_argument('-l', '--left', default='/HDD/tjh/msl/left_nav_pair/10500.jpg',
                    help="left image's path or numpy data")
parser.add_argument('-r', '--right', default='/HDD/tjh/msl/right_nav_pair/10500.jpg',
                    help="right image's path or numpy data")
parser.add_argument('-d', '--max_disparity', default=100, type=int, help="max disparity")
parser.add_argument('-s', '--shift', default=1, type=int, help="max vertical shift(one side)")
parser.add_argument('-b', '--block_size', type=int, default=3, help='block size')
parser.add_argument('-t', '--threshold', default=1, type=int,
                    help="[0-255], the lower, the weaker limit")
parser.add_argument('-p', '--penalty', default=2, type=float,
                    help="the coefficient of cost at the situation of beyond threshold")
parser.add_argument('-D', '--device', default='cpu', type=str, help="'cpu' or 'cuda'")
parser.add_argument('-P', '--enable_parallel', default=False, type=bool,
                    help="parallel compute, True or False")
parser.add_argument('-S', '--enable_save', default=False, action='store_true',
                    help="save the output as image if specified")
parser.add_argument('--save_path', default='./', type=str, help="path to save output image")
parser.add_argument('-R', '--resize', default=1, type=int,
                    help="resize the image as given ratio before processing, range: (0, 1)")
parser.add_argument('-B', '--blur', default=0, type=float,
                    help="blur the image pair before processing."
                         "Actually, the value is SIGMA in Gaussian Blur. The bigger, the fuzzier")
parse = parser.parse_args()

print("WARNING: This file should running on linux, the output image's name shall wrong in Windows system!")


class MatchingSAD:
    def __init__(self, left=parse.left, right=parse.right, max_disparity=parse.max_disparity,
                 shift=parse.shift, block_size=parse.block_size, threshold=parse.threshold,
                 penalty=parse.penalty, device=parse.device, enable_parallel=parse.enable_parallel,
                 enable_save=parse.enable_save, save_path=parse.save_path, resize=parse.resize, blur=parse.blur):
        """Binocular stereo matching using SAD

        :param left: left image's path or numpy data
        :param right: right image's path or numpy data
        :param max_disparity: max disparity
        :param shift: max vertical pixel shift (one side)
        :param block_size: block size
        :param threshold: [0-255], the lower, the weaker limit
        :param penalty: the coefficient of cost at the situation of beyond threshold
        :param device: 'cpu' or 'cuda'
        :param enable_parallel: enable parallel computing, valid for cpu and gpu
        :param enable_save: saving the output as image file
        :param save_path: saving path
        :param resize: resize the image as given ratio before processing, range: (0, 1),
            WARNING: all the other parameters should be specified w.r.t the resized image
        :param blur: blur the image pair before processing.
            Actually, the value is SIGMA in Gaussian Blur. The bigger, the fuzzier

        :type left: str or numpy
        :type right: str or numpy
        :type max_disparity: int
        :type shift: int
        :type block_size: int
        :type threshold: int
        :type penalty: float
        :type device: str
        :type enable_parallel: bool
        :type enable_save: bool
        :type save_path: str
        :type resize: float
        :type blur: float
        """

        self.left = left
        self.right = right
        self.max_disparity = max_disparity
        self.shift = shift
        self.block_size = block_size
        self.threshold = threshold
        self.penalty = penalty
        self.device = device
        self.enable_parallel = enable_parallel
        self.enable_save = enable_save
        self.save_path = save_path
        self.resize = resize
        self.blur = blur

        # device judgement
        if not (self.device == 'cpu' or self.device == 'cuda'):
            raise ValueError("device type error")
        if self.device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        # penalty judgement
        if self.penalty <= 0:
            raise ValueError("penalty should be a positive number")

        # image judgement
        if type(self.left) == str:
            '''save the image's name and load the image if parameter left is a string(path)'''
            if not (os.path.exists(self.left) and os.path.exists(self.right)):
                raise ValueError("image's path is not valid")
            self.name = shell('./get_name.sh', self.left)

            self.left_img_pre = cv2.imread(self.left, cv2.IMREAD_GRAYSCALE)
            self.right_img_pre = cv2.imread(self.right, cv2.IMREAD_GRAYSCALE)

            self.left_img_pre = torch.from_numpy(self.left_img_pre
                                                 ).to(device=self.device, dtype=torch.float)
            self.right_img_pre = torch.from_numpy(self.right_img_pre
                                                  ).to(device=self.device, dtype=torch.float)
        else:
            '''load images if numpy data'''
            if not (type(self.left) == np.ndarray and type(self.right) == np.ndarray):
                raise ValueError("image's type is not 'ndarray'")
            self.left_img_pre = torch.from_numpy(self.left
                                                 ).to(device=self.device, dtype=torch.float)
            self.right_img_pre = torch.from_numpy(self.right
                                                  ).to(device=self.device, dtype=torch.float)

        # resize judgement
        if (self.resize * min(self.left_img_pre.shape)) <= self.block_size \
                or self.resize > 1:
            raise ValueError("resize ratio is out of bound")
        else:
            '''resize images'''
            stride = round(1 / self.resize)
            weight = torch.tensor([[[[1.]]]], device=self.device)

            self.left_img = F.conv2d(input=self.left_img_pre.unsqueeze(0).unsqueeze(0),
                                     weight=weight, stride=stride)
            self.right_img = F.conv2d(input=self.right_img_pre.unsqueeze(0).unsqueeze(0),
                                      weight=weight, stride=stride)

        # image shape after resize before padding
        self.img_shape = tuple(self.left_img.shape[2:])

        # list of image pixels
        # length: img_shape[0] * img_shape[1]
        self.img_pixels = list()
        for row in range(self.img_shape[0]):
            for col in range(self.img_shape[1]):
                self.img_pixels.append((row, col))

        # max_disparity judgement
        if (self.max_disparity < 0) or (self.max_disparity >= (self.img_shape[1] - 1)):
            raise ValueError("max disparity out of bound")

        # shift judgement
        if self.shift < 0 or self.shift > (self.img_shape[0] / 2):
            raise ValueError("vertical shift out of bound or too big")

        # block size judgement
        if self.block_size <= 0 or self.block_size >= min(self.img_shape):
            raise ValueError("block size out of bound")
        if self.block_size % 2 == 0:
            raise ValueError("block size should be an odd")
        else:
            '''reflect padding'''

            self.padding = block_size // 2
            temp = nn.ReflectionPad2d(self.padding)
            self.left_img = temp(self.left_img)
            self.right_img = temp(self.right_img)

        # threshold judgement
        if self.threshold < 0 or self.threshold > 255:
            raise ValueError("threshold out of bound")

        # save path judgement
        if self.enable_save:
            if not os.path.exists(self.save_path):
                raise ValueError("saving path is not valid")

        # blur judgement
        if self.blur < 0 or (6 * self.blur + 1) >= min(self.img_shape):
            raise ValueError("the value of blur is out of bound")
        elif self.blur is not 0:
            '''blur the images after resize'''

            temp = GaussianBlur(self.left_img, kernel_size=odd(self.blur * 6 + 1),
                                sigma=self.blur, device=self.device)
            self.left_img = temp.output()

            temp = GaussianBlur(self.right_img, kernel_size=odd(self.blur * 6 + 1),
                                sigma=self.blur, device=self.device)
            self.right_img = temp.output()

        # pure-one convolution
        temp = Conv(self.block_size, self.device)
        self.left_img = temp(self.left_img)
        self.right_img = temp(self.right_img)

        # start matching
        if self.enable_parallel:
            self.disparity = self._parallel1()
        else:
            self.disparity = self._solo0()

        if self.enable_save:
            self.save()

    def save(self):
        try:
            cv2.imwrite(self.save_path + self.name + '.jpg',
                        self.disparity)
        except IOError:
            print("disparity wrote failed")
        else:
            print("disparity wrote success")

    def _parallel0(self):
        """parallel version of solo1
        threads: total pixel numbers
        not work for cuda
        """

        dict_disparity = dict()
        left_img = self.left_img.squeeze()
        right_img = self.right_img.squeeze()
        penalty = torch.tensor(self.penalty, device=self.device)
        cost_max = torch.tensor(256 * (self.block_size ** 2), device=self.device)
        cost_threshold = torch.tensor(self.threshold * (self.block_size ** 2), device=self.device)

        def job(left_px, idx):
            pixels = search_pixels(left_px=left_px, max_disparity=self.max_disparity,
                                   shift=self.shift, img_shape=self.img_shape)
            flag = torch.tensor(left_px[1])
            cost = cost_max.clone().detach()
            for px in pixels:
                cost_ = (left_img[left_px] - right_img[px]).abs()
                if cost_ <= cost_threshold:
                    cost_ *= penalty
                if cost_ < cost:
                    cost = cost_
                    flag = px[1]
            dict_disparity[idx] = left_px[1] - flag

        for p in range(self.img_shape[0] * self.img_shape[1]):
            if self.device == torch.device('cuda'):
                mp.set_start_method('spawn')
            row = p // self.img_shape[1]
            col = p % self.img_shape[1]
            temp = mp.Process(target=job, args=((row, col), p))
            temp.start()

        dict_disparity = dict(sorted(dict_disparity.items(),
                                     key=lambda item: item[0])
                              )
        disparity = list(dict_disparity.values())
        disparity = torch.tensor(disparity, dtype=torch.uint8
                                 ).reshape(self.img_shape[0], -1).numpy()
        return disparity

    def _parallel1(self):
        """parallel version of solo0
        threads: same number of image's height
        works well on cpu, but not on gpu
        """

        left_img = self.left_img.squeeze()
        right_img = self.right_img.squeeze()
        penalty = torch.tensor(self.penalty, device=self.device)
        cost_max = torch.tensor(255 * (self.block_size ** 2), device=self.device)
        cost_threshold = torch.tensor(self.threshold * (self.block_size ** 2), device=self.device)

        def job(q, left_row_pixels, idx):
            """child process for mp.Process

            :param q: mp.Queue, for the row-like disparities' storing
            :param left_row_pixels: a list of the left image's pixels in one row
            :param idx: index flag, for "q"'s sort in post process
            :return: disparity of the input row
            """
            row_disparity = list()
            for left_px in left_row_pixels:
                pixels = search_pixels(left_px=left_px, max_disparity=self.max_disparity,
                                       shift=self.shift, img_shape=self.img_shape)
                # flag: store the up-to-date matched pixel's column
                flag = torch.tensor(left_px[1])
                cost = cost_max.clone().detach()

                for px in pixels:
                    """find the matched pixel's column"""
                    cost_ = (left_img[left_px] - right_img[px]).abs()
                    if cost_ <= cost_threshold:
                        cost_ *= penalty
                    if cost_ < cost:
                        cost = cost_
                        flag = px[1]
                row_disparity.append(left_px[1] - flag)
            q.put((idx, row_disparity))

        # start multiprocessing for left image's rows
        if __name__ == '__main__':
            if self.device == torch.device('cuda'):
                mp.set_start_method('spawn')
            q = mp.Queue()
            for row in range(self.img_shape[0]):
                left_row_pixels = self.img_pixels[row * self.img_shape[1]:
                                                  (row + 1) * self.img_shape[1] - 1]
                temp = mp.Process(target=job, args=(q, left_row_pixels, row))
                temp.start()

            # extract (idx, row) pairs from 'q' into a list
            idx_row = list()
            for i in range(self.img_shape[0]):
                idx_row.append(q.get())

            idx_row = dict(sorted(idx_row, key=lambda pair: pair[0]))
            disparity_ = list(idx_row.values())
            disparity = list()
            for row in disparity_:
                disparity += row
            disparity = torch.tensor(disparity, dtype=torch.uint8
                                     ).reshape(self.img_shape[0], -1).numpy()
            return disparity

    def _solo0(self):
        """non-parallel matching, just using 'for' loops"""

        disparity = list()
        left_img = self.left_img.squeeze()
        right_img = self.right_img.squeeze()
        penalty = torch.tensor(self.penalty, device=self.device)
        cost_max = torch.tensor(255 * (self.block_size ** 2), device=self.device)
        cost_threshold = torch.tensor(self.threshold * (self.block_size ** 2), device=self.device)

        for row in tqdm.tqdm(range(self.img_shape[0])):
            for col in range(self.img_shape[1]):
                pixels = search_pixels(left_px=(row, col), max_disparity=self.max_disparity,
                                       shift=self.shift, img_shape=self.img_shape)
                flag = torch.tensor(col)
                cost = cost_max.clone().detach()
                for px in pixels:
                    cost_ = (left_img[row, col] - right_img[px]).abs()
                    if cost_ <= cost_threshold:
                        cost_ *= penalty
                    if cost_ < cost:
                        cost = cost_
                        flag = px[1]
                disparity.append(col - flag)

        disparity = torch.tensor(disparity, dtype=torch.uint8
                                 ).reshape(self.img_shape[0], -1).numpy()
        return disparity


class Conv(nn.Module):
    """for pure-one convolution"""

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


class _GaussianBlurNet(nn.Module):
    def __init__(self, kernel, device):
        """

        :param kernel: from cv2.getGaussianKernel
        :param device: device
        :type kernel: 1-dim list
        :type device: torch.device

        :return: blurred image
        :rtype: same as input image(tensor[1,1,H,W] or numpy)
        """
        super(_GaussianBlurNet, self).__init__()
        self.train(False)
        self.kernel = kernel
        self.device = device

        # 两个一维卷积核
        # 可以不使用nn.Parameter，因为该方法将v/hkernel转化为可训练的参数，但这里根本不用训练
        vkernel = torch.from_numpy(self.kernel).to(device=self.device, dtype=torch.float)
        hkernel = vkernel.T
        self._vkernel = vkernel.unsqueeze(0).unsqueeze(0)
        self._hkernel = hkernel.unsqueeze(0).unsqueeze(0)
        self._padding = len(self.kernel) // 2

    # 两次卷积实现分离高斯模糊
    def forward(self, img):
        if not torch.is_tensor(img):
            img = torch.from_numpy(img).to(device=self.device, dtype=torch.float)
            img = img.unsqueeze(0).unsqueeze(0)

        temp = nn.ReflectionPad2d((0, 0, self._padding, self._padding))
        img = F.conv2d(temp(img), weight=self._vkernel)
        temp = nn.ReflectionPad2d((self._padding, self._padding, 0, 0))
        img = F.conv2d(temp(img), weight=self._hkernel)

        if not torch.is_tensor(img):
            img = img.squeeze().to(device='cpu', dtype=torch.uint8).numpy()

        return img


class GaussianBlur:
    def __init__(self, img, kernel_size, sigma, device=torch.device('cpu')):
        """image blur using spilt gaussian kernel

        :param img: input image
        :param kernel_size: gaussian kernel size
        :param sigma: gaussian blur's sigma
        :param device: compute device

        :type
        """

        # un-annotation the following lines if their is no odd kernel pre-check
        # # 核心非奇数返回Error
        # if not (kernel_size % 2):
        #     raise ValueError("kernel size must be and odd")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.img = img
        self.kernel = cv2.getGaussianKernel(kernel_size, sigma)
        net = _GaussianBlurNet(self.kernel, device)
        self.out = net(self.img)

    def output(self):
        return self.out


def shell(path, param):
    """run a bash script and return the output

    :param path: script's path
    :param param: script's parameter
    :type path: str
    :type param: str
    :return: script's output
    :rtype: str
    """

    cmd = "sh -e %s %s" % (path, param)
    cmd = shlex.split(cmd)
    output = subprocess.run(cmd, capture_output=True)
    return output.stdout.strip().decode('utf-8')


def odd(num):
    """return the closest odd number

    :param num: python number
    :type num: int or float
    :rtype: int
    """

    num = int(num)
    num = num if num % 2 == 1 else num + 1
    return num


def search_pixels(left_px, max_disparity, shift, img_shape):
    """定位匹配范围的左上角和右下角坐标

    :param left_px: left image's pixel waiting for match
    :param max_disparity: max_disparity
    :param shift: vertical shift
    :param img_shape: image's shape
    :type left_px: tuple or list
    :type max_disparity: int
    :type shift: int
    :type img_shape: list or tuple

    :return: a list of potential pixels in right image
    :rtype: list of tuples
    """

    # get computational left-top and right-bottom corner first
    left_top_corner = [left_px[0] - shift,
                       left_px[1] - max_disparity]
    right_bottom_corner = [left_px[0] + shift,
                           left_px[1]]

    # then judge their validation
    if left_top_corner[0] < 0:
        left_top_corner[0] = 0
    if left_top_corner[1] < 0:
        left_top_corner[1] = 0
    if right_bottom_corner[0] >= img_shape[0]:
        right_bottom_corner[0] = img_shape[0] - 1

    # push pixel
    pixels = []
    for row in range(left_top_corner[0], right_bottom_corner[0] + 1):
        for col in range(left_top_corner[1], right_bottom_corner[1] + 1):
            pixels.append((row, col))
    return pixels


if __name__ == '__main__':
    MatchingSAD(enable_save=True, resize=0.50, blur=0, max_disparity=100, shift=0, block_size=3,
                left='/HDD/tjh/SceneFlow/sampler/Driving/RGB_cleanpass/left/0400.png',
                right='/HDD/tjh/SceneFlow/sampler/Driving/RGB_cleanpass/right/0400.png',
                enable_parallel=True, device='cpu')
