#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import xml.dom.minidom
from xml.dom.minidom import Document  
import math
import codecs
from tqdm import tqdm
import numpy as np
import random
from PIL import Image
from glob import glob
from collections import Counter
import h5py

import sys
# In[7]:


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
from torchsummary import summary
from utils1 import progress_bar
#from models.SENet.se_resnet import se_resnet20_v2,se_resnet20
from utils.LabelSmooth import LabelSmoothing

# from models.preact_resnet import *
#from models.preact_resnet_sn import *
from models.senet import SENet18,SENet34,SENet101,SENet152
from models.SENet.se_resnet import se_resnet18,se_resnet34,se_resnet50
# from models.senet_sn import SENet34_SN,SENet101_SN,SENet152_SN
#多gpu训练




from imgaug import augmenters as iaa

seq = iaa.Sequential([iaa.GaussianBlur(sigma=(0.25, 0.75)),
#     iaa.Fliplr(0.5),#有向右转，向左转，不能这么做
#   iaa.Flipud(0.2),
#     iaa.Sometimes(0.1,iaa.Rot90((1, 3))),#90,180,270
    iaa.Affine(
        scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
        translate_percent={"x": (-0.0, 0.0), "y": (-0.01, 0.01)},
        rotate=(-1.1, 1.1),
        shear=(-1.1, 1.1)
    ),
#     iaa.Multiply((0.9, 1.1), per_channel=False),
    iaa.SomeOf(1 ,[
    iaa.PiecewiseAffine(scale=(0.01,0.02)),#0.01   
    iaa.PerspectiveTransform(scale=(0.01, 0.03))], random_order=True)# 透视变化，值越大，变化越明显
]) # apply augmenters in random modelorder
    


# In[10]:


class Generator():
    def __init__(self,
                 filepath='./data/',
                 batch_size=8,
                 datatype='train',
                 split=0.1):
        train_file = h5py.File(os.path.join(filepath, 'hld_train_aug.h5'), 'r')     
        self.train_X = train_file['images']
        self.train_Y = train_file['labels']
        # 统计每一个数据集的数量
        self.num_train = self.train_Y.shape[0]
        # 按照batch_size进行（分组）采样
        # 得到每一个分组的索引 [0, 8, 16, 24, ...]
        #num_groups = int((self.num_train + self.num_val) / batch_size)
        num_groups = int((self.num_train) / batch_size)
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

        images = self.train_X[idx:idx + self.batch_size,:,:,:]

        labels = self.train_Y[idx:idx + self.batch_size]

        self.index += 1
        if self.index >= self.count:
            self.index = 0
            np.random.shuffle(self.indices)

        images = np.asarray(images, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)

        if self.datatype== 'train':
            return seq.augment_images(images), labels
        else:
            return images, labels

    def __next__(self):
        return self.next_batch()


# In[11]:


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


# In[12]:


def train_model(model, criterion, optimizer, num_epochs=25,batch_size = 32):

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
            
            trainfalse = {x: 0 for x in np.arange(16)} 
            trainall = trainfalse#每一类epoch的统计

            valfalse = {x: 0 for x in np.arange(16)} 
            valall = valfalse

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
                #print(labels.shape)
                #print(labels)
                #labels =labels.argmax(dim=1)#CE默认不支持one-hot编码
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
                #print(labels.data.cpu().numpy())
                
                print_traincorrects += torch.sum(preds == labels.data)
                
                ###########################
                falselenth=len(labels.data[np.where((preds != labels.data).cpu())])
                if phase=='train':
                    for j in range(falselenth):
                        trainfalse[labels.data[np.where((preds != labels.data).cpu())].cpu().numpy()[j]]+=1
                    # for i in range(batch_size):
                    #     trainall[labels.data.cpu().numpy()[i]]+=1

                if phase=='val':
                    for k in range(falselenth):
                        valfalse[labels.data[np.where((preds != labels.data).cpu())].cpu().numpy()[k]]+=1                
                    # for i in range(batch_size):
                    #     valall[labels.data.cpu().numpy()[i]]+=1
               # print_iter_train=1000 #每1000输出一次train的loss和acc

                #if i%print_iter_train==0 and i>0 and phase=='train':
                 #   print('\r{} Loss: {:.4f} Acc: {:.4f}'.format(phase,print_trainloss/(print_iter_train*data.batch_size), float(print_traincorrects)/(data.batch_size*print_iter_train)))
                  #  print_traincorrects=0.0
                  #  print_trainloss=0.0   
                if i>0 and phase=='train':
                    #print(i)    
                    progress_bar(i, Iter, 'Loss:%.2f| Acc: %.2f%%(%d/%d) | LR:%.4f'
            % (running_loss/(i+1), 100.*float(running_corrects)/total, running_corrects, total, batch_lr))

            epoch_loss = float(running_loss) / (Iter*data.batch_size)
            epoch_acc = float(running_corrects) / (Iter*data.batch_size)
#######################################################################          
            acc_ygx = 0
            # if phase=='train':
            #     for i in range(16):
            #         if trainall[i]==0:
                        
            #             acc_ygx += float(trainall[i] - trainfalse[i])/trainall[i]
            # if phase=='val':
            #     for i in range(16):
            #         if valall[i]==0:

            #             acc_ygx += float(valall[i] - valfalse[i])/valall[i]
#########################################################33
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
                  
            if phase=='train':
                print('TrainFalse: {}'.format(trainfalse))
            if phase=='val':
                print('ValFalse: {}'.format(valfalse))
                
            save_path=save_root+args.model
            if not os.path.isdir(save_path): 
                os.mkdir(save_path)

            torch.save(model.state_dict(),save_path+'/'+args.model+'_{0}.pth'.format(epoch+51))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    
    return model


# In[14]:


###############################################
parser = argparse.ArgumentParser(description='PyTorch LCZ Training')
parser.add_argument('--lr_period', default=10, type=float, help='learning rate schedule restart period')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')#0.02
parser.add_argument('--model', '-s', default='SENet34', help='saves state_dict on every epoch (for resuming best performing model and saving it)')

save_root='/home/zj/senetial/save_models/'

args = parser.parse_args()


device_ids = [0]
use_gpu = torch.cuda.is_available()
lr_trace = []
#定义网络

#model = PreActResNet18_SN()
#model =SENet101_SN()
model =se_resnet18(16)

if use_gpu and len(device_ids)>1:#多gpu训练
    model = model.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)
    torch.backends.cudnn.benchmark = True

if use_gpu and len(device_ids)==1:#单gpu训练
    model = model.cuda()
    torch.backends.cudnn.benchmark = True
print(model)

model.load_state_dict(torch.load('/home/zj/senetial/save_models/SENet34/SENet34_50.pth'))
#定义损失函数

#w = torch.tensor([1.0,1.0,1.5,1.5,5.0,5.0,5.0,5.0,8.0,5.0,2.0,5.0,5.0,3.0,5.0,1.0])
w = torch.tensor([0.8,0.8,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.8])
criterion = nn.CrossEntropyLoss(weight = w)
#criterion = LabelSmoothing(size=17,smoothing=0.1)
criterion.cuda()
#定义优化器
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)

#定义trian的数据和val的数据
print('Preparing data..')
batch_size = 32
trian_data = Generator(datatype='train',batch_size=batch_size,split=0.2)
val_data = Generator(datatype='val',batch_size=batch_size,split=0.2)

model_ft = train_model(model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           num_epochs=50,
                           batch_size = batch_size)


# In[ ]:




