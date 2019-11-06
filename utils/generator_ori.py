# -*- coding: utf-8 -*-

import h5py
import os

import numpy as np





class generator_ori():
    def __init__(self,
                 filepath='/home/zj/senetial/data/training.h5',
                 batch_size=8,
                 datatype='train'):
        if datatype == 'val':
            filepath='/home/zj/senetial/data/validation.h5'   
        if datatype == 'train':
            filepath='/home/zj/senetial/data/training.h5' 
        data_file = h5py.File(filepath, 'r')
        self.X1 = data_file['sen1']
        self.X2 = data_file['sen2']
        self.Y = data_file['label']

        # 按照batch_size进行（分组）采样
        # 得到每一个分组的索引 [0, 8, 16, 24, ...]
        self.indices = np.arange(self.Y.shape[0] // batch_size) * batch_size
        np.random.seed(1)
        np.random.shuffle(self.indices)
        # 这里只选择总数的后1/10作为验证集
        # 其余的作为训练集
        self.count = self.indices.size
        self.batch_size = batch_size
        self.index = 0

    def next_batch(self, batch_size=8):
        idx = self.indices[self.index]
        images_1 = self.X1[idx:idx + batch_size]
        images_2 = self.X2[idx:idx + batch_size]
        labels = self.Y[idx:idx + batch_size]

        self.index += 1
        if self.index >= self.count:
            self.index = 0
            np.random.shuffle(self.indices)

        images_1 = np.asarray(images_1, dtype=np.float32)
        images_2 = np.asarray(images_2, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)

        return [images_1, images_2], labels

    def __next__(self):
        return self.next_batch(self.batch_size)
