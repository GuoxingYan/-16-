# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:33:44 2018
求mean和val，这里的设置np.random.seed(2)和 batch_size=126666,正好弄出253332-381323含有train的1/4 valtest全部
@author: ygx
"""

import h5py
import numpy as np
import os
import time


#多gpu训练
device_ids = [0,1]

def AddFeatures(array):
    """
    输入一个18个波段的concat之后的图像NHWC
    NDVI、SI阴影指数、ndbi建筑物指数
    """
    ndvi=(array[:,:,:,14]-array[:,:,:,11])/(array[:,:,:,14]+array[:,:,:,11])
    si=(array[:,:,:,14]+array[:,:,:,9]+array[:,:,:,11]+array[:,:,:,10])/4
    ndbi=(array[:,:,:,17]-array[:,:,:,14])/(array[:,:,:,17]+array[:,:,:,14])
    output=np.concatenate((array,ndvi[:,:,:,np.newaxis],si[:,:,:,np.newaxis],ndbi[:,:,:,np.newaxis]),axis=3)
    return output

#381323 253332 - 253332+126666
class Generator():
    def __init__(self,
                 filepath='/home/zj/senetial/data',
                 batch_size=9000,
                 datatype='train',
                 split=0.1):
        train_file = h5py.File(os.path.join(filepath, 'training.h5'), 'r')
        val_file = h5py.File(os.path.join(filepath, 'valtest_meanstd.h5'), 'r')

        self.train_X1 = train_file['sen1']
        self.train_X2 = train_file['sen2']

        self.val_X1 = val_file['sen1']
        self.val_X2 = val_file['sen2']

        # 统计每一个数据集的数量
        self.num_train = self.train_X1.shape[0]
        self.num_val = self.val_X2.shape[0]
        # 按照batch_size进行（分组）采样
        # 得到每一个分组的索引 [0, 8, 16, 24, ...]
        num_groups = int((self.num_train + self.num_val) / batch_size)
        self.indices = np.arange(num_groups) * batch_size#这里其实少了最后面的一个batch，但是就不用考虑train和val中间的情况
        #np.random.seed(1)
        np.random.seed(2)
        np.random.shuffle(self.indices)
        # 这里只选择总数的后1/10作为验证集
        # 其余的作为训练集
        print self.indices
        split = int(num_groups * split)
        split = -split if split else None
        if datatype == 'train':
            self.indices = self.indices[:split]
        else:
            self.indices = self.indices[split:]
        #count是指示的总数/batchsize
        self.count = self.indices.size
        self.batch_size = batch_size
        self.index = 0

    def next_batch(self, batch_size=16):
        idx = self.indices[self.index]
        if idx > self.num_train:
            idx = idx - self.num_train
            X1 = self.val_X1
            X2 = self.val_X2
            #Y = self.val_Y
        else:
            X1 = self.train_X1
            X2 = self.train_X2
            #Y = self.train_Y
        images_1 = X1[idx:idx + batch_size]
        images_2 = X2[idx:idx + batch_size]
        #labels = Y[idx:idx + batch_size]

        self.index += 1
        if self.index >= self.count:
            self.index = 0
            np.random.shuffle(self.indices)

        images_1 = np.asarray(images_1, dtype=np.float32)
        images_2 = np.asarray(images_2, dtype=np.float32)#(352366, 32, 32, 10)
        #labels = np.asarray(labels, dtype=np.float32)
        
        images=np.concatenate((images_1,images_2),axis=3)#合并归一化操作#(352366, 32, 32, 18)
        
        images = AddFeatures(images)        
        #images=RandomPaddingCrop(RandomFlip(images))
        
        return images

    def __next__(self):
        return self.next_batch(self.batch_size)
        

#####Test        

#test_file = h5py.File('/home/zj/senetial/data/round1_test_a_20181109.h5', 'r')
#test_file = h5py.File('/home/zj/senetial/data/validation.h5','r')
#test_sen1 = test_file['sen1']
#test_sen2 = test_file['sen2']
#count=test_sen1.shape[0]

#测试一下时间
start=time.time()

#计算方11011
mean,std=0,0
 #前30000个均值，方差
#train
train = Generator(filepath='/home/zj/senetial/data',
                 batch_size=126666,
                 datatype='trainval',
                 split=0.1)
datas_train= train.__next__()#一次跑出batchsize张 32
mean =np.mean(datas_train,axis=(0,1,2))
std=np.std(datas_train,axis=(0,1,2))
####test
#images=(np.concatenate((test_sen1[:,:,:,:],test_sen2[:,:,:,:]),axis=3))#列表：可以自动识别是否到了最后
#print images.shape
#mean =np.mean(images,axis=(0,1,2))
#std=np.std(images,axis=(0,1,2))
#    
print '_'*60
print mean
print '_'*60
print std 
print '_'*60
end=time.time()
print end-start #100次1.14秒
