# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:33:44 2018
原先愚蠢的认为都读到内存再打乱，可惜了h5文件太大了
@author: ygx
"""
from __future__ import division
import h5py
import numpy as np
#import matplotlib.pyplot as plt
import os
import time

import torch
from torch import optim
from torch.autograd import Variable
#import torchvision
from torchvision import transforms
#from torch.utils.data import Dataset
import torch.nn as nn

from tqdm import tqdm
#from focal_loss import FocalLoss
#from models.resnet_me import ResNet
from models.resnet_v2_sn import resnetv2sn18
from models.TSnet import TSnet

from data_generotor import Generator
#多gpu训练
device_ids = [0,1]
#由于torch中搞得transform.totensor不支持3维以上的变成到0-1之间，所以需要重新定义函数
def to01(array):
    """
    变成0-1之间，平衡各个变量之间的影响因素
    但是我提前知道那个变量影响大，是不是意味着可以从这里做手脚！
    input:np.array [1,32,32,*]
    """
    min_value=np.min(array,axis=(0,1,2))[np.newaxis,:]#增加维度，也可以用np.expand_dims
    max_value=np.max(array,axis=(0,1,2))[np.newaxis,:]
    output=(array-min_value)/(max_value-min_value)
    return output
#def MNDWI(array):
#    
#    return mdnwi
#
#def NDVI(array):
#    ndvi=(array[:,14,:,:]-array[:,10,:])/(array[:,14,:,:]+array[:,10,:])
#    return ndvi
#    
#    
#def bandfeature(array):
#    """
#    输入一个18个波段的concat之后的图像
#    得到：NDVI、NDISI、BI/MDNWI/FVC
#    """
#   
#    
#    return output


class generotor_batch():

    def __init__(self,
                 base_dir='/home/zj/senetial/data',
                 batch_size=32,
                 data_aug=False,#暂时没写
                 data_set='train'):
                 #split=0.068):
        ##写入只需要执行一次的操作。
        self.batch_size=batch_size
        self.data_set=data_set
        path_train = os.path.join(base_dir,'training.h5')
        path_val = os.path.join(base_dir,'validation.h5')
        
        fid_train = h5py.File(path_train,'r')
        fid_val = h5py.File(path_val,'r')
        fid_test = h5py.File(os.path.join(base_dir,'round1_test_a_20181109.h5'))
        if self.data_set=='train':#注意只可以正向序列访问，即s1[[1,2,3],:,:,:]但是s1[[3，2],:,:,:]就不行，所以随机打乱然后按照打乱的顺序，从头来时读取是不可行的     
            self.s1=fid_train['sen1']   #(352366, 32, 32, 8)
            self.s2=fid_train['sen2']   #(352366, 32, 32, 10)
            self.label=fid_train['label']  #(352366, 17)
            self.count=self.s1.shape[0]
        elif self.data_set=='val':  
            self.s1=fid_val['sen1']      
            self.s2 = fid_val['sen2']   #(24119, 32, 32, 10)
            self.label = fid_val['label']     #(24119, 17)
            self.count=self.s2.shape[0]
        else:
            self.s1=fid_test['sen1']      
            self.s2 = fid_test['sen2']   #(24119, 32, 32, 10)
            self.count=self.s2.shape[0]
             
###输入前16000，验证下效果        
        #self.count=16000
        
        np.random.seed(0)
        self.permulation = np.random.permutation(self.count)  
        self.index=0#用于存放目前的图片索引
        #把下面这几句话从next中拿出来，省了一半的时间。
        self.image_s1=np.empty([self.batch_size,32,32,8])
        self.image_s2=np.empty([self.batch_size,32,32,10])
        self.labels=np.empty([self.batch_size,17])
        
        
    def next_batch(self):
        #for i in tqdm(range(self.batch_size)):
        for i in range(self.batch_size):
            self.index += 1 
            self.index = self.index % self.count  #1%300008=1
            if self.index == 0:
                #跑完一个轮回，重新打乱
                self.permulation = np.random.permutation(self.count)
                self.index=1
            if self.data_set=='train' or self.data_set=='val':
                self.image_s1[i,:,:,:]=np.expand_dims(np.array(self.s1[self.permulation[self.index-1],:,:,:]),axis=0)
                self.image_s2[i,:,:,:]=np.expand_dims(np.array(self.s2[self.permulation[self.index-1],:,:,:]),axis=0)
                self.labels[i,:]=np.expand_dims(np.array(self.label[self.permulation[self.index-1],:]),axis=0)
            else:
                self.image_s1[i,:,:,:]=np.expand_dims(np.array(self.s1[self.index-1,:,:,:]),axis=0)#不进行打乱操作
                self.image_s2[i,:,:,:]=np.expand_dims(np.array(self.s2[self.index-1,:,:,:]),axis=0)
            
        #images=(np.concatenate((self.image_s1,self.image_s2),axis=3))
        images=to01(np.concatenate((self.image_s1,self.image_s2),axis=3))#合并归一化操作
        if self.data_set=='train' or self.data_set=='val':
            return images,self.labels
        else:
            return images
        
    def __next__(self):
        return self.next_batch()
         
#测试一下时间
start=time.time()
train = generotor_batch()
#计算方11011
mean,std=0,0

for i in (range(300)):
 #前30000个均值，方差
   datas_train,_= train.__next__()#一次跑出batchsize张 32
#   mean =np.mean(datas_train,axis=(0,1,2))
#   std=np.std(datas_train,axis=(0,1,2))
#   print '_'*60
#   print mean
#   print '_'*60
#   print std 
end=time.time()
print end-start #100次1.14秒
