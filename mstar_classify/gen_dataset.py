# -*- coding: utf-8 -*-

import PIL
from PIL import Image
import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_txt_path, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        f = open(data_txt_path, 'r')
        imgs = []
        for line in f:
            line = line.strip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fp, label = self.imgs[index]
        img = Image.open(fp)
        if len(img.split()) > 1:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)