from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    """conv2d + BN """

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    """Conv3d + BN"""

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
                         nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    """basic ResNet

    convbn + relu + convbn
    expansion = 1, which tend to expand output layers
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        """

        :param inplanes:
        :param planes:
        :param stride:
        :param downsample:
        :param pad:
        :param dilation:
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))
        '''convbn + relu'''

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)
        '''convbn'''

        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left  = F.pad(torch.index_select(left,  3, Variable(torch.LongTensor([i for i in range(shift, width)])).cuda()),(shift,0,0,0))
        shifted_right = F.pad(torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width-shift)])).cuda()),(shift,0,0,0))
        out = torch.cat((shifted_left,shifted_right),1).view(batch,filters*2,1,height,width)
        return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1)
        return out


class feature_extraction(nn.Module):
    """CNN + SPP module in paper"""
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32

        # conv0 in paper
        # equal to 7 * 7 conv's receptive field, but deeper
        self.firstconv = nn.Sequential(convbn(in_planes=3, out_planes=32, kernel_size=3, stride=2, pad=1, dilation=1),
                                       nn.ReLU(inplace=True),   # 'inplace' save memory
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, planes=32, blocks=3, stride=1, pad=1, dilation=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)     # 1 -> 2
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 4)     # 2 -> 4

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        """for "downsample = None", return a ResNet which contains 3 layers basic ResNet

        :param block: basic ResNet
        :param planes: output planes
        :param blocks: the number of ResNet layers
        :param stride:
        :param pad:
        :param dilation:
        :return:
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # three 3*3 conv layers
        output = self.firstconv(x)
        # four ResNet blocks
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        # four scale SPP
        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, size=(output_skip.shape[2],
                                                             output_skip.shape[3]),
                                       mode='bilinear', align_corners=False)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, size=(output_skip.shape[2],
                                                             output_skip.shape[3]),
                                       mode='bilinear', align_corners=False)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, size=(output_skip.shape[2],
                                                             output_skip.shape[3]),
                                       mode='bilinear', align_corners=False)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, size=(output_skip.shape[2],
                                                             output_skip.shape[3]),
                                       mode='bilinear', align_corners=False)

        output_feature = torch.cat((output_raw, output_skip, output_branch4,
                                    output_branch3, output_branch2, output_branch1),
                                   1)
        output_feature = self.lastconv(output_feature)

        return output_feature



