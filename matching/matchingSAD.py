"""SAD Matching for non-calibrated image pairs

IDEA before coding:
    class API with variety methods, including:
        device choice(using torch, anyway);
        parallel switch(to cpu and gpu, so totally 4 choices, but first achieve non-parallel with cpu)
        command-line saving and showing support(argparse: --show, --write)
        GET support (for other module)
        down-sampling options and gaussian blur options (both for command-line convenience)

    some important steps:
        reflect padding in pre-processing

    MODULE LIKE CODING !!!!!!
        for better debug

TODO:
    MAX DISPARITY may has a linear variation, try to take this in consideration
    using "edge detection" or "gaussian blur" to estimate MAX DISPARITY ??? then to fit the linear MAX DISPARITY line
        which is a parallel task(gpu accelerate)
    adaptable block size(considering the texture complexity)
    parameter VERTICAL SHIFT may be estimated through feature extraction??
    various penalties
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

parser = argparse.ArgumentParser(description="SAD Matching for non-calibrated image pairs")
parser.add_argument('-l', '--left', default='/HDD/tjh/msl/left_nav_pair/525.jpg',
                    help="left image's path or numpy data")
parser.add_argument('-r', '--right', default='/HDD/tjh/msl/right_nav_pair/525.jpg',
                    help="right image's path or numpy data")
parser.add_argument('-d', '--max_disparity', default=200, type=int, help="max disparity")
parser.add_argument('-s', '--shift', default=1, type=int, help="max vertical shift(one side)")
parser.add_argument('-b', '--block_size', type=int, default=5, help='block size')
parser.add_argument('-t', '--threshold', default=1, type=int,
                    help="[0-255], the lower, the weaker limit")
parser.add_argument('-D', '--device', default='cpu', type=str, help="'cpu' or 'gpu'")
parser.add_argument('-p', '--enable_parallel', default=False, type=bool,
                    help="parallel compute, True or False")
parser.add_argument('-S', '--enable_save', default=False, action='store_true',
                    help="save the output as image if specified")
parser.add_argument('--save_path', default='./', type=str, help="path to save output image")
parser.add_argument('-R', '--resize', default=1, type=int,
                    help="resize the image as given ratio before processing")
parser.add_argument('-B', '--blur', default=0, type=float,
                    help="blur the image pair before processing."
                         "Actually, the value is SIGMA in Gaussian Blur. The bigger, the fuzzier")
parse = parser.parse_args()

print("WARNING: This file should running in linux environment, the output image's name shall wrong in Windows system!")


class MatchingSAD:
    def __init__(self, left=parse.left, right=parse.right, max_disparity=parse.max_disparity,
                 shift=parse.shift, block_size=parse.block_size, threshold=parse.threshold,
                 device=parse.device, enable_parallel=parse.enable_parallel, enable_save=parse.enable_save,
                 save_path=parse.save_path, resize=parse.resize, blur=parse.blur):
        """Binocular stereo matching using SAD

        :param left: left image's path or numpy data
        :param right: right image's path or numpy data
        :param max_disparity: max disparity
        :param shift: max vertical pixel shift (one side)
        :param block_size: block size
        :param threshold: [0-255], the lower, the weaker limit
        :param device: 'cpu' or 'gpu'
        :param enable_parallel: enable parallel computing, valid for cpu and gpu
        :param enable_save: saving the output as image file
        :param save_path: saving path
        :param resize: resize the image as given ratio before processing
        :param blur: blur the image pair before processing.
            Actually, the value is SIGMA in Gaussian Blur. The bigger, the fuzzier
        :type left: str or numpy
        :type right: str or numpy
        :type max_disparity: int
        :type shift: int
        :type block_size: int
        :type threshold: int
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
        self.device = device
        self.enable_parallel = enable_parallel
        self.enable_save = enable_save
        self.save_path = save_path
        self.resize = resize
        self.blur = blur

        if type(self.left) == str:
            '''save the image's name and load the image if parameter left is a string(path)'''
            if not (os.path.exists(self.left) and os.path.exists(self.right)):
                raise ValueError("image's path is not valid")
            self.name = shell('./get_name.sh', self.left)
            self.left_img = cv2.imread(self.left, cv2.IMREAD_GRAYSCALE)
            self.right_img = cv2.imread(self.right, cv2.IMREAD_GRAYSCALE)
        else:
            if not (type(self.left) == np.ndarray and type(self.right) == np.ndarray):
                raise ValueError("image's type is not 'ndarray'")
            self.left_img = self.left
            self.right_img = self.right


        if self.resize * min(self.left.shape) <= self.block_size:
            raise ValueError("resize ratio is too small")
        self.imgH, self.imgW = self.left_img.shape
        '''image's height and width'''

        if self.max_disparity < 0 or self.max_disparity >= (self.imgW - 1):
            raise ValueError("max disparity out of bound")
        if self.shift < 0 or self.shift > (self.imgH / 2):
            raise ValueError("vertical shift out of bound or too big")
        if self.block_size <= 0 or self.block_size >= min(self.imgW, self.imgH):
            raise ValueError("block size out of bound")
        if self.threshold < 0 or self.threshold > 255:
            raise ValueError("threshold out of bound")
        if not (self.device == 'cpu' or self.device == 'gpu'):
            raise ValueError("device type error")
        if self.enable_save:
            if not os.path.exists(self.save_path):
                raise ValueError("saving path is not valid")






def shell(path, param):
    """run a bash script and return the output

    :param path: script's path
    :param param: script's parameter
    :type path: str
    :type param: str
    :return: script's output
    :rtype: str
    """
    import subprocess
    import shlex

    cmd = "sh -e %s %s" % (path, param)
    cmd = shlex.split(cmd)
    output = subprocess.run(cmd, capture_output=True)

    return output.stdout.strip().decode('utf-8')

