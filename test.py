# -*- coding: utf-8 -*-


"""
Created on Tue Dec  4 17:33:44 2018
原先愚蠢的认为都读到内存再打乱，可惜了h5文件太大了
@author: ygx
"""

import h5py
import numpy as np
#import matplotlib.pyplot as plt
import os
import torch

from torch.autograd import Variable
#import torchvision
from torchvision import transforms
#from torch.utils.data import Dataset
import torch.nn as nn
import math
from tqdm import tqdm
#from focal_loss import FocalLoss
#from models.resnet_me import ResNet
from models.SENet.se_resnet import se_resnet20_v2
#tsnet+sn+se+labelsmooth

#tsnet+sn+se+crossenproty

from models.senet import SENet18,SENet34
from models.senet_sn import SENet34_SN,SENet101_SN,SENet152_SN
#def mean_std(array):
#    mean=[-3.76171612e-04,-2.40010091e-05,-6.96983124e-05, 1.22778358e-04,
#          4.86409500e-02, 2.79232573e-01,-1.83128277e-03, 2.96897620e-03,
#          1.27696816e-01, 1.14926075e-01, 1.11185264e-01, 1.23210661e-01,
#          1.64488456e-01, 1.86060429e-01, 1.79203636e-01, 2.00056916e-01,
#          1.72947497e-01, 1.28166367e-01]
#    std=[0.18249301,0.1829685, 0.46908406,0.46917832,0.59569964,3.92665674,
#         0.86806065,0.6843909, 0.03525861,0.04047221,0.05539265,0.05115125,
#         0.06193356,0.07342109,0.07623389,0.0829271,0.08878885,0.08166186]
#    return (array-mean)/std

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

def mean_std(array):
    mean=[-5.8652742e-05,2.1940528e-05,1.0698503e-05,-3.6694932e-05,
          2.8304448e-02,1.8809982e-01,7.6072977e-04,1.0500627e-03,
          4.5042928e-02,4.6203922e-02,5.0576344e-02,5.2854732e-02,
          7.6116115e-02,8.3651222e-02,8.3404467e-02,8.4830634e-02,
          8.3397724e-02,5.9738528e-02,1.2501612e-01,5.3771760e-02,-1.6543813e-01]
    std=[0.15613347,0.15583488,0.423213,0.41939119, 2.4466505 , 8.333362,
         2.254153,1.3736475, 0.0687826,  0.06539123, 0.07428898, 0.07567783,
         0.09257834, 0.10948441, 0.10750295, 0.12227393, 0.10479108, 0.08892089,
         0.20687017, 0.07645338, 0.2877441 ]
    return (array-mean)/std
       
use_gpu = torch.cuda.is_available()

device_ids=[0]        
#TEST
model = SENet101_SN()  
#model =TSnetSE()
#model =TSnetSECE()



if use_gpu and len(device_ids)>1:#多gpu训练
    model = model.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)
if use_gpu and len(device_ids)==1:#单gpu训练
    model = model.cuda()
    
print(model)
#加载数据

test_file = h5py.File('/home/zj/senetial/data/round2_test_a_20190121.h5', 'r')
test_sen1 = test_file['sen1']
test_sen2 = test_file['sen2']
count=test_sen1.shape[0]
print(count)

#加载模型
pretained_model = torch.load('/home/zj/senetial/save_models/SENet101_SN/SENet101_SN_7.pth')
model.load_state_dict(pretained_model)

batchsize=128
#保存结果
f=open('/home/zj/senetial/data/SENet101_SNA_07_0.9805.csv','w')
for i in tqdm(range(int(math.ceil(float(count)/batchsize)))):###不用管这里到底能不能整除，你进行向上取整就好
    onehot = np.zeros((batchsize,17),dtype=int)
    images=mean_std(AddFeatures(np.concatenate((test_sen1[i*batchsize:(i+1)*batchsize,:,:,:],test_sen2[i*batchsize:(i+1)*batchsize,:,:,:]),axis=3)))#列表：可以自动识别是否到了最后
    inputs = Variable((torch.from_numpy(images)).float().permute(0, 3, 1, 2).cuda(device_ids[0]))
    #print model
    outputs = model(inputs)
    scores, preds = torch.max(outputs.data, 1)#preds
    for i in range(inputs.shape[0]):#注意这里不能用batchsize，只能用列表[:]后的shape
        onehot[i,preds[i]]=1
        f.writelines(','.join(map(str,onehot[i]))+'\n')
        #f.writelines(','.join(map(str,onehot[i]))+','+str(scores[i])+'\n')

        
    
