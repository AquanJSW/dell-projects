import subprocess
import shlex
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# class DownSampling(nn.Module):
#     """down sampling"""
#     def __init__(self, ratio):
#         """
#
#         :param ratio: down sampling ratio
#         :type ratio: float
#         """
#         super(DownSampling, self).__init__()
#         self.ratio = ratio
#         self.weight = torch.tensor([[[[1.]]]])
#         self.stride = round(1 / ratio)
#
#     def forward(self, x):
#         x = F.conv2d(x.to(dtype=torch.float).unsqueeze(0).unsqueeze(0), self.weight, stride=self.stride)
#         return x
