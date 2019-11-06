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

import torch
from torch import optim
from torch.autograd import Variable
#import torchvision
#from torchvision import transforms
#from torch.utils.data import Dataset
import torch.nn as nn

from tqdm import tqdm

from models.TSNet.TSnet import TSnet
from models.SENet.se_resnet import se_resnet20_v2,se_resnet20
from utils.LabelSmooth import LabelSmoothing
#多gpu训练
device_ids = [0]
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
                 filepath='/home/ygx/zj/sentinel/data',
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
        np.random.seed(2)
        np.random.shuffle(self.indices)
        # 这里只选择总数的后1/10作为验证集
        # 其余的作为训练集
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
        images_2 = np.asarray(images_2, dtype=np.float32)#(352366, 32, 32, 10)
        labels = np.asarray(labels, dtype=np.float32)
        images=np.concatenate((images_1,images_2),axis=3)#合并归一化操作#(352366, 32, 32, 18)
        
        images = mean_std(AddFeatures(images))  
        #images=RandomPaddingCrop(RandomFlip(images))
        images=RandomPaddingCrop(RandomPre(images))
        
        return images, labels

    def __next__(self):
        return self.next_batch(self.batch_size)
        
class Generator_ori():
    def __init__(self,
                 filepath='/home/ygx/zj/sentinel/data/training.h5',
                 batch_size=8,
                 datatype='train'):
        if datatype == 'val':
            filepath='/home/ygx/zj/sentinel/data/validation.h5'   
        if datatype == 'train':
            filepath='/home/ygx/zj/sentinel/data/training.h5' 
        data_file = h5py.File(filepath, 'r')
        self.X1 = data_file['sen1']
        self.X2 = data_file['sen2']
        self.Y = data_file['label']
        self.nums=self.Y.shape[0]
        # 按照batch_size进行（分组）采样
        # 得到每一个分组的索引 [0, 8, 16, 24, ...]
        self.indices = np.arange(self.Y.shape[0] // batch_size) * batch_size
        np.random.seed(2)
        np.random.shuffle(self.indices)
        # 这里只选择总数的后1/10作为验证集
        # 其余的作为训练集
        self.count = self.indices.size
        self.batch_size = batch_size
        self.index = 0
        

    def next_batch(self, batch_size=8):
        idx = self.indices[self.index]
        images_1 = self.X1[idx:min(self.nums,idx + batch_size)]
        images_2 = self.X2[idx:min(self.nums,idx + batch_size)]
        labels = self.Y[idx:min(self.nums,idx + batch_size)]

        self.index += 1
        if self.index >= self.count:
            self.index = 0
            np.random.shuffle(self.indices)

        images_1 = np.asarray(images_1, dtype=np.float32)
        images_2 = np.asarray(images_2, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        images=np.concatenate((images_1,images_2),axis=3)#合并归一化操作#(352366, 32, 32, 18)
        
        images = mean_std(AddFeatures(images))        
        images=RandomPaddingCrop(RandomPre(images))
        
        return images, labels

    def __next__(self):
        return self.next_batch(self.batch_size)
        
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        #开始第几次循环
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase!!!
        for phase in ['train', 'val']:
            #根据phase不同，将读入的data不同，然后传入
            if phase == 'train':
                data = trian_data
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                data =val_data
                model.train(False)  # Set model to evaluate mode
                #model.eval() #等效于model.train(False)仅仅当模型中有Dropout和BatchNorm是才会有影响。

            running_loss = 0.0
            running_corrects = 0
            print_trainloss=0.0
            print_traincorrects=0.0

            # Iterate over data.
            #Iter=int(data.s1.shape[0]/float(data.batch_size))
###输入前16000，验证下效果              
            #Iter=1000
            Iter=int(data.count)
         
            for i in tqdm(range(Iter)):#用1.6w张图片看下效果
                # get the inputs   
                inputs, labels = data.next_batch()#迭代器 
                inputs=np.ascontiguousarray(inputs, dtype=np.float32)
                # wrap them in Variable
                if use_gpu:#np_>FloatTensor_>Variable
                    inputs = Variable((torch.from_numpy(inputs)).float().permute(0, 3, 1, 2).cuda(device_ids[0]))#输入必须是float N C H W
                    #inputs = train_transform(inputs)
                    labels = Variable((torch.from_numpy(labels)).long().cuda(device_ids[0]))#label必须是long
                else:
                    inputs, labels = Variable(torch.from_numpy(inputs).permute(0, 3, 1, 2)), Variable(torch.from_numpy(labels).long())
                    #inputs = train_transform(inputs)
                # zero the parameter gradients 因为本身是累加的
                optimizer.zero_grad()

                # forward  CE
                outputs = model(inputs)
                labels =labels.argmax(dim=1)#CE默认不支持one-hot编码
                _, preds = torch.max(outputs.data, 1)#这里已经转成了
                loss = criterion(outputs, labels)#CE默认不支持one-hot编码
                 # forward  label Smoothing
#                outputs = model(inputs)
#                labels =labels.argmax(dim=1)#
#                _, preds = torch.max(outputs.data, 1)#输入不是one-hot
#                loss = criterion(outputs.log(), labels)

                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step() 
                    #optimizer.module.step()
                # statistics
                running_loss += loss.data[0]
                print_trainloss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                print_traincorrects += torch.sum(preds == labels.data)
                print_iter_train=4000 #每1000输出一次train的loss和acc
                print_iter_val=5000#每3000输出一次val的acc
                if i%print_iter_train==0 and i>0 and phase=='train':
                    print('\r{} Loss: {:.4f} Acc: {:.4f}'.format(phase,print_trainloss/(print_iter_train*data.batch_size), float(print_traincorrects)/(data.batch_size*print_iter_train)))
                    print_traincorrects=0.0
                    print_trainloss=0.0   
                    
#                if i%print_iter_val==0 and  i>0 :
#                    print_valcorrects=0.0
#                    for i in tqdm(range(int(val_data.count))):  
#                        inputs_val, labels_val = val_data.next_batch()#迭代器  
#                        inputs_val=np.ascontiguousarray(inputs_val, dtype=np.float32)
#                        inputs_val = Variable((torch.from_numpy(inputs_val)).float().permute(0, 3, 1, 2).cuda())#输入必须是float N C H W
#                        #inputs = train_transform(inputs)
#                        labels_val = Variable((torch.from_numpy(labels_val)).long().cuda())#label必须是long
#                        outputs_val = model(inputs_val)
#                        labels_val =labels_val.argmax(dim=1)#CE默认不支持one-hot编码
#                        _, preds_val = torch.max(outputs_val.data, 1)#这里已经转成了
#                        print_valcorrects += torch.sum(preds_val == labels_val.data)
#                        val_acc=float(print_valcorrects)/(print_iter_val*val_data.batch_size)
#                    print('-' * 10)
#                    print('\rVal Acc: {:.4f}'.format(val_acc))#\r表示从头开始
#                    print('-' * 10)
                    
            epoch_loss = float(running_loss) / (Iter*data.batch_size)
            epoch_acc = float(running_corrects) / (Iter*data.batch_size)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 如果val的acc大于之前最好的val_acc
#            if phase == 'val' and epoch_acc > best_acc:
#                best_acc = epoch_acc
#                best_model_wts = model.state_dict()
            ## 保存模型
            torch.save(model.state_dict(),'/home/ygx/zj/sentinel/save_models/SENet/SEnet_model_{0}.pth'.format(epoch+26))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights 最后循环完，加入最好的module
    #model.load_state_dict(best_model_wts)
    
    return model

###############################################
use_gpu = torch.cuda.is_available()
#定义网络
#model=TSnet(17)
model = se_resnet20_v2()

model = model.cuda()
print model

model.load_state_dict(torch.load('/home/ygx/zj/sentinel/save_models/SENet/SEnet_model_25.pth'))
#定义损失函数sentinel
#criterion = nn.CrossEntropyLoss()
weight = torch.FloatTensor([0.014382772458182685, 0.0693341582332007, 0.08994341111230936, 0.024551176901290137, 
                            0.04680644557079854, 0.10015154697104715, 0.009277285549684135, 0.11160554650562199, 
                            0.03855082499446598, 0.03392495303179081, 0.12175408524091427, 0.0270003348790746, 
                            0.026009887446575435, 0.11742619889546664, 0.006788396156269334, 0.022414194332029763, 0.14007878172127844])
criterion = LabelSmoothing(size=17,smoothing=0.01)
criterion.cuda(device_ids[0])
#定义优化器
#optimizer=optim.Adam(model.parameters(), lr=0.000005, betas=(0.9, 0.999))

optimizer = optim.SGD(model.parameters(), lr=0.000005, momentum=0.9,weight_decay=0.0005)

#定义优化策略
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,50)
#定义trian的数据和val的数据
trian_data = Generator(filepath='/home/ygx/zj/sentinel/data',datatype='train',batch_size=16,split=0.088)
val_data = Generator(filepath='/home/ygx/zj/sentinel/data',datatype='val',batch_size=16,split=0.088)
#trian_data = Generator_ori(filepath='/home/ygx/zj/sentinel/data',batch_size=16,datatype='train')
#val_data = Generator_ori(filepath='/home/ygx/zj/sentinel/data',batch_size=16,datatype='val')

model_ft = train_model(model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           num_epochs=30)



##TEST
#model_best = torch.load('/home/ygx/zj/sentinel/save_models/model_1.pth')
#test_data = generotor_batch(batch_size=16,data_set='test')
#inputs = Variable((torch.from_numpy(test_data)).float().permute(0, 3, 1, 2).cuda())
#pre = model_best(inputs)
#pre_hot=torch.FloatTensor(inputs.shape[0],17)
#pre_hot.zero_()
#f=open('','w')
#for i in range(pre.shape[0]):
#    pre_hot[i,pre[i]]=1
#    f.write(pre_hot[i,:]+'\n')









