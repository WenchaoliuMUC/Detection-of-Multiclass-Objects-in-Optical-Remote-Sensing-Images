#!/usr/bin/python
# encoding: utf-8
import torch
from torch.utils.data import Dataset
from utils import read_truths_args
from image import *


class listDataset(Dataset):

    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=24):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if self.train:
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)
        else:
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
    
            labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            label = torch.zeros(800*5)
            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 5.0/img.width, self.shape).astype('float32'))
            except Exception:
                tmp = torch.zeros(1, 5)
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            if tsz > 800*5:
                label = tmp[0:800*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return (img, label)
