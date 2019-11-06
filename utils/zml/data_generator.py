import h5py
import os

import numpy as np


class generator():
    def __init__(self,
                 filepath='/home/lmzwhu/programs/DATAS/LCZ/training.h5',
                 batch_size=8,
                 datatype='train',
                 split=0.1):
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
        split = round(self.Y.shape[0] // batch_size * split)
        split = -split if split else None
        if datatype == 'train':
            self.indices = self.indices[:split]
        else:
            self.indices = self.indices[split:]

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


class MixedGenerator():
    def __init__(self,
                 filepath='/home/lmzwhu/programs/DATAS/LCZ/',
                 batch_size=8,
                 datatype='train',
                 split=0.1):
        train_file = h5py.File(os.path.join(filepath, 'training.h5'), 'r')
        val_file = h5py.File(os.path.join(filepath, 'validation.h5'), 'r')

        self.train_X1 = train_file['sen1']
        self.train_X2 = train_file['sen2']
        self.train_Y = train_file['label']

        self.val_X1 = val_file['sen1']
        self.val_X2 = val_file['sen2']
        self.val_Y = val_file['label']

        # 统计每一个数据集的数量
        self.num_train = self.train_Y.shape[0]
        self.num_val = self.val_Y.shape[0]
        # 按照batch_size进行（分组）采样
        # 得到每一个分组的索引 [0, 8, 16, 24, ...]
        num_groups = (self.num_train + self.num_val) // batch_size
        self.indices = np.arange(num_groups) * batch_size
        np.random.seed(1)
        np.random.shuffle(self.indices)
        # 这里只选择总数的后1/10作为验证集
        # 其余的作为训练集
        split = round(num_groups * split)
        split = -split if split else None
        if datatype == 'train':
            self.indices = self.indices[:split]
        else:
            self.indices = self.indices[split:]

        self.count = self.indices.size
        self.batch_size = batch_size
        self.index = 0

    def next_batch(self, batch_size=8):
        idx = self.indices[self.index]
        if idx > self.num_train:
            idx = idx - self.num_train
            X1 = self.val_X1
            X2 = self.val_X2
            Y = self.val_Y
        else:
            X1 = self.train_X1
            X2 = self.train_X2
            Y = self.train_Y
        images_1 = X1[idx:idx + batch_size]
        images_2 = X2[idx:idx + batch_size]
        labels = Y[idx:idx + batch_size]

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
