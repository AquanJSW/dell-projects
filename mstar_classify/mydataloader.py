# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchvision.transforms as transforms
from gen_dataset import MyDataset
import copy


def mydataloader():
    batch_size = 16
    testset_partial = 0.2   # 测试集的占比
    shuffle = True  # 是否打乱数据集(Sample阶段，非DataLoader的shuffle)
    random_seed = 25

    class_dict = {'2S1': 0, 'BMP2': 1, 'BRDM2': 2, 'BTR60': 3, 'BTR70': 4,
                  'D7': 5, 'T62': 6, 'T72': 7, 'ZIL131': 8, 'ZSU234': 9}

    classes = ('2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70',
               'D7', 'T62', 'T72', 'ZIL131', 'ZSU234')

    PATH = '/home/tjh/projects/py/mstar_classify/'

    # 注意transform.Compose 内的各个操作的顺序，不正确可能会引发错误
    transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    for c in range(10):
        # 乱序
        txt_path = '%s%s.txt' % (PATH, classes[c])
        dataset = MyDataset(data_txt_path=txt_path, transform=transform)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        dataset_split = int(np.floor(testset_partial * dataset_size))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        testset_indices, trainset_indices = indices[:dataset_split], indices[dataset_split:]

        # 建立trainset, testset的堆栈
        # 注意dataset不能直接copy给新变量，会共用空间，要用到copy模块
        if c == 0:
            trainset = copy.deepcopy(dataset)
            testset = copy.deepcopy(dataset)
            trainset.imgs = []
            testset.imgs = []

        for i in trainset_indices:
            trainset.imgs.append(dataset.imgs[i])

        for i in testset_indices:
            testset.imgs.append(dataset.imgs[i])


    # 整体shuffle
    if shuffle:
        trainset_size = len(trainset)
        testset_size = len(testset)
        trainset_indices = list(range(trainset_size))
        testset_indices = list(range(testset_size))
        np.random.shuffle(trainset_indices)
        np.random.shuffle(testset_indices)

        train_sampler = torch.utils.data.SubsetRandomSampler(trainset_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(testset_indices)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   sampler=train_sampler, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  sampler=test_sampler, num_workers=2)
        return trainloader, testloader


if __name__ == '__main__':
    mydataloader()
