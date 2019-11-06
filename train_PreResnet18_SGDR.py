# -*- coding: utf-8 -*-

from __future__ import division

"""
Created on Tue Dec  4 17:33:44 2018
原先愚蠢的认为都读到内存再打乱，可惜了h5文件太大了
@author: ygx
"""

import h5py
import numpy as np
#import matplotlib.pyplot as plt
import os
import time
import argparse
import torch
from torch import optim
from torch.autograd import Variable
#import torchvision
#from torchvision import transforms
#from torch.utils.data import Dataset
import torch.nn as nn
import math
from tqdm import tqdm
from utils1 import progress_bar
#from models.SENet.se_resnet import se_resnet20_v2,se_resnet20
from utils.LabelSmooth import LabelSmoothing

# from models.preact_resnet import *
#from models.preact_resnet_sn import *
from models.senet import SENet18,SENet34,SENet101,SENet152
from models.senet_sn import se_resnet18,se_resnet34,se_resnet50
#多gpu训练

#由于torch中搞得transform.totensor不支持3维以上的变成到0-1之间，所以需要重新定义函数
 
#    
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

def RandomHorizontalFlip(array):
    a1 = np.random.choice(a=[0,1], size=1)
    if a1==1:
        return array[:,:,:,::-1]
    else:
        return array

def RandomVerticalFlip(array):
    a1 = np.random.choice(a=[0,1], size=1)
    if a1==1:
        array=array.transpose(0,3,1,2)#因为地址连续的原因，必须加copy
        array=array[:,:,::-1,::-1]
        #return np.ascontiguousarray(array.transpose(0,2,3,1),dtype=np.float32)
        return array.transpose(0,2,3,1)
    else:
        return array
        
def RandomFlip(array):
    return RandomVerticalFlip(RandomHorizontalFlip(array))
    
def RandomRotate180(array):
    a1 = np.random.choice(a=[0,1], size=1)
    if a1==1:
        array=array.transpose(0,3,1,2)#因为地址连续的原因，必须加copy
        array=array[:,::-1,::-1,::-1]
        return array.transpose(0,2,3,1)
    else:
        return array

def RandomRotate90(array):
    a1 = np.random.choice(a=[0,1], size=1)
    if a1==1:
        return array.transpose(0,2,1,3)
    else:
        return array

def RandomRotate(array):
    return RandomRotate180(RandomRotate90(array))

def RandomPre(array):
    return RandomRotate(RandomFlip(array))

    
def Padding(array,filters=0,nums_side=4):
    """默认以0为填充，四个边都添加4个0 NHWC"""
    N,H,W,C=array.shape
    output=np.zeros((N,H+2*nums_side,W+2*nums_side,C))
    output[:,nums_side:H+nums_side,nums_side:W+nums_side,:]=array
    return output

def RandomPaddingCrop(array,filters=0,nums_side=4,size=(32,32)):
    """一半的概率padding之后裁剪成原图大小 NHWC"""
    a1 = np.random.choice(a=[0,1], size=1)
    if a1==1:
        PaddingArray=Padding(array,filters=0,nums_side=4)
        start_w,start_h=np.random.choice(8,2)#从[0:8]随机去出两个数字
        return PaddingArray[:,start_w:start_w+size[0],start_h:start_h+size[1],:]
    else:
        return array
    
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

class Generator():
    def __init__(self,
                 filepath='/home/zj/senetial/data',
                 batch_size=256,
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
        num_groups = int((self.num_train + self.num_val) / batch_size)
        self.indices = np.arange(num_groups) * batch_size#这里其实少了最后面的一个batch，但是就不用考虑train和val中间的情况
        np.random.seed(3)
        np.random.shuffle(self.indices)
        # 这里只选择总数的后1/10作为验证集
        # 其余的作为训练集 cudnn.benchmark = True
        split = int(num_groups * split)
        split = -split if split else None
        self.datatype=datatype
        if datatype == 'train':
            self.indices = self.indices[:split]
        else:
            self.indices = self.indices[split:]
        #count是指示的总数/batchsize
        self.count = self.indices.size
        self.batch_size = batch_size
        self.index = 0

    def next_batch(self):
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
        images_1 = X1[idx:idx + self.batch_size]
        images_2 = X2[idx:idx +self. batch_size]
        labels = Y[idx:idx + self.batch_size]

        self.index += 1
        if self.index >= self.count:
            self.index = 0
            np.random.shuffle(self.indices)

        images_1 = np.asarray(images_1, dtype=np.float32)
        images_2 = np.asarray(images_2, dtype=np.float32)#(352366, 32, 32, 10)
        labels = np.asarray(labels, dtype=np.float32)
        images=np.concatenate((images_1,images_2),axis=3)#合并归一化操作#(352366, 32, 32, 18)
        images = mean_std(AddFeatures(images)) 
        if self.datatype== 'train':
            images=(RandomPre(images))
            return images, labels
        else:
            return images, labels

    def __next__(self):
        return self.next_batch()
        
def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def sgdr(period, batch_idx):
    # returns normalised anytime sgdr schedule given period and batch_idx
    # best performing settings reported in paper are T_0 = 10, T_mult=2
    # so always use T_mult=2
    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx/restart_period > 1.:
        batch_idx = batch_idx - restart_period
        restart_period = restart_period * 2.

    radians = math.pi*(batch_idx/restart_period)
    return 0.5*(1.0 + math.cos(radians))

        
def train_model(model, criterion, optimizer, num_epochs=25):

    since = time.time() 

    best_acc = 0.0

    for epoch in range(num_epochs):
        #global optimizer
        #开始第几次循环
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        

        # Each epoch has a training and validation phase!!!
        for phase in ['train','val']:
            #根据phase不同，将读入的data不同，然后传入
            if phase == 'train':
                data = trian_data
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                data =val_data
                model.train(False)  # Set model to evaluate mode
                #model.eval() #等效于model.train(False)仅仅当模型中有Dropout和BatchNorm是才会有影响。

            running_loss = 0.0
            running_corrects = 0
            print_trainloss=0.0
            print_traincorrects=0.0
            total = 0

            # Iterate over data.
            #Iter=int(data.s1.shape[0]/float(data.batch_size))
###输入前16000，验证下效果              
            #Iter=1000
            Iter=int(data.count)
         

	
     
            start_batch_idx=Iter*epoch
            
            trainfalse = {x: 0 for x in np.arange(17)} 
            valfalse = {x: 0 for x in np.arange(17)} 
            lr_period = args.lr_period*Iter
            for i in (range(Iter)):#用1.6w张图片看下效果
                # get the inputs   
                inputs, labels = data.next_batch()#迭代器 
                inputs=np.ascontiguousarray(inputs, dtype=np.float32)
                # wrap them in Variable
                if use_gpu:#np_>FloatTensor_>Variable
                    inputs = Variable((torch.from_numpy(inputs)).float().permute(0, 3, 1, 2).cuda())#输入必须是float N C H W
                    #inputs = train_transform(inputs)
                    labels = Variable((torch.from_numpy(labels)).long().cuda())#label必须是long
                else:
                    inputs, labels = Variable(torch.from_numpy(inputs).permute(0, 3, 1, 2)), Variable(torch.from_numpy(labels).long())

                total += labels.size(0)
                global_step = i+start_batch_idx
                
                batch_lr = args.lr*sgdr(lr_period, global_step)
                lr_trace.append(batch_lr)
                optimizer = set_optimizer_lr(optimizer, batch_lr)
                # zero the parameter gradients 因为本身是累加的   
                optimizer.zero_grad()

                # forward  CE
                outputs = model(inputs)
                labels =labels.argmax(dim=1)#CE默认不支持one-hot编码
                _, preds = torch.max(outputs.data, 1)#这里已经转成了
                loss = criterion(outputs, labels)#CE默认不支持one-hot编码
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step() 

                # statistics
                running_loss += loss.item()
                print_trainloss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                print_traincorrects += torch.sum(preds == labels.data)
                
                ###########################
                falselenth=len(labels.data[np.where((preds != labels.data).cpu())])
                if phase=='train':
                    for j in range(falselenth):
                         trainfalse[labels.data[np.where((preds != labels.data).cpu())].cpu().numpy()[j]]+=1
                if phase=='val':
                    for k in range(falselenth):
                         valfalse[labels.data[np.where((preds != labels.data).cpu())].cpu().numpy()[k]]+=1                
                
               # print_iter_train=1000 #每1000输出一次train的loss和acc

                #if i%print_iter_train==0 and i>0 and phase=='train':
                 #   print('\r{} Loss: {:.4f} Acc: {:.4f}'.format(phase,print_trainloss/(print_iter_train*data.batch_size), float(print_traincorrects)/(data.batch_size*print_iter_train)))
                  #  print_traincorrects=0.0
                  #  print_trainloss=0.0   
                if i>0 and phase=='train':    
                    progress_bar(i, Iter, 'Loss:%.2f| Acc: %.2f%%(%d/%d) | LR:%.4f'
            % (running_loss/(i+1), 100.*float(running_corrects)/total, running_corrects, total, batch_lr))

            epoch_loss = float(running_loss) / (Iter*data.batch_size)
            epoch_acc = float(running_corrects) / (Iter*data.batch_size)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
                  
            if phase=='train':
                print('TrainFalse: {}'.format(trainfalse))
            if phase=='val':
                print('ValFalse: {}'.format(valfalse))
                
            save_path=save_root+args.model
            if not os.path.isdir(save_path): 
                os.mkdir(save_path)

            torch.save(model.state_dict(),save_path+'/'+args.model+'_{0}.pth'.format(epoch+1))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def sgdr(period, batch_idx):
    # returns normalised anytime sgdr schedule given period and batch_idx
    # best performing settings reported in paper are T_0 = 10, T_mult=2
    # so always use T_mult=2
    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx/restart_period > 1.:
        batch_idx = batch_idx - restart_period
        restart_period = restart_period * 2.

    radians = math.pi*(batch_idx/restart_period)
    return 0.5*(1.0 + math.cos(radians))

    return model

###############################################
parser = argparse.ArgumentParser(description='PyTorch LCZ Training')
parser.add_argument('--lr_period', default=10, type=float, help='learning rate schedule restart period')
parser.add_argument('--lr', default=0.004, type=float, help='learning rate')#0.02
parser.add_argument('--model', '-s', default='SENet101_SN', help='saves state_dict on every epoch (for resuming best performing model and saving it)')

save_root='/home/zj/senetial/save_models/'

args = parser.parse_args()


device_ids = [0]
use_gpu = torch.cuda.is_available()
lr_trace = []
#定义网络

#model = PreActResNet18_SN()
model =se_resnet18(num_classes=16)

if use_gpu and len(device_ids)>1:#多gpu训练
    model = model.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)
    torch.backends.cudnn.benchmark = True

if use_gpu and len(device_ids)==1:#单gpu训练
    model = model.cuda()
    torch.backends.cudnn.benchmark = True
print(model)

model.load_state_dict(torch.load('/home/zj/senetial/save_models/SENet101_SN/ori/SENet101_SN_29.pth'))
#定义损失函数
criterion = nn.CrossEntropyLoss()
#criterion = LabelSmoothing(size=17,smoothing=0.1)
criterion.cuda()
#定义优化器
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.8,weight_decay=5e-4)

#定义trian的数据和val的数据
print('Preparing data..')
trian_data = Generator(filepath='/home/zj/senetial/data',datatype='train',batch_size=128,split=0.2)
val_data = Generator(filepath='/home/zj/senetial/data',datatype='val',batch_size=128,split=0.2)

model_ft = train_model(model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           num_epochs=100)










