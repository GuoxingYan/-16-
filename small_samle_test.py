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
from torchvision import transforms
#from torch.utils.data import Dataset
import torch.nn as nn

from tqdm import tqdm
#from focal_loss import FocalLoss
#from models.resnet_me import ResNet
#from models.resnet.resnet_v2_sn import resnetv2sn18
from models.TSNet.TSnet import TSnet

#多gpu训练
device_ids = [0]
#由于torch中搞得transform.totensor不支持3维以上的变成到0-1之间，所以需要重新定义函数
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


##      

#train_transform =transforms.Compose([transforms.Resize(42),
#                                     transforms.RandomHorizontalFlip(0.5),
#                                     transforms.RandomVerticalFlip(0.5),
#                                     transforms.RandomCrop(32)])                                              
#def to01(array):
#    """
#    变成-1-1之间，平衡各个变量之间的影响因素
#    但是我提前知道那个变量影响大，是不是意味着可以从这里做手脚！
#    input:np.array [1,32,32,*]
#    """
#    min_value=np.min(array,axis=(0,1,2))[np.newaxis,:]#增加维度，也可以用np.expand_dims
#    max_value=np.max(array,axis=(0,1,2))[np.newaxis,:]
#    output=(array-min_value)/(max_value-min_value)
#    return output
def mean_std(array):
    mean=[-2.1557720e-05,-1.5458829e-05,1.2666348e-04,9.4661758e-05,
          2.8482638e-02,1.8647610e-01,4.7995898e-04,1.3979858e-03,
          4.4616148e-02,4.5875672e-02,5.0444003e-02,5.2574083e-02,
          7.6114096e-02,8.2751110e-02,8.2533732e-02,8.3922490e-02,8.2764462e-02,5.9669822e-02]
    std=[0.15752424,0.15763956,0.42342424,0.41814595,2.777361,8.940067,2.7208624, 
         1.484532,  0.06893244,0.0655082, 0.07437851,0.0757706,0.0926284, 
         0.10997296,0.10793569,0.12283914,0.10512207,0.08932678]
    return (array-mean)/std


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

class Generator():
    def __init__(self,
                 filepath='/home/zj/senetial/data',
                 batch_size=16,
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
        #np.random.seed(1)
        np.random.seed(2)
        np.random.shuffle(self.indices)
        # 这里只选择总数的后1/10作为验证集
        # 其余的作为训练集from models.resnet_v2_sn import resnetv2sn18
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
        images_2 = np.asarray(images_2, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        images=mean_std(np.concatenate((images_1,images_2),axis=3))#合并归一化操作

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
        #for phase in ['val']:
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
                # wrap them in Variable
                if use_gpu:#np_>FloatTensor_>Variable
                    inputs = Variable((torch.from_numpy(inputs)).float().permute(0, 3, 1, 2).cuda())#输入必须是float N C H W
                    #inputs = train_transform(inputs)
                    labels = Variable((torch.from_numpy(labels)).long().cuda())#label必须是long
                else:
                    inputs, labels = Variable(torch.from_numpy(inputs).permute(0, 3, 1, 2)), Variable(torch.from_numpy(labels).long())
                    #inputs = train_transform(inputs)
                # zero the parameter gradients 因为本身是累加的
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                labels =labels.argmax(dim=1)#CE默认不支持one-hot编码
                _, preds = torch.max(outputs.data, 1)#这里已经转成了
                loss = criterion(outputs, labels)#CE默认不支持one-hot编码
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step() 
                # statistics
                running_loss += loss.data[0]
                print_trainloss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                print_traincorrects += torch.sum(preds == labels.data)
                print_iter_train=100 #每1000输出一次train的loss和acc
                print_iter_val=5000#每3000输出一次val的acc
                if i%print_iter_train==0 and i>0 and phase=='train':
                    print('\r{} Loss: {:.4f} Acc: {:.4f} lr:{:.4f}'.format(phase,print_trainloss/(print_iter_train*data.batch_size), 
                          float(print_traincorrects)/(data.batch_size*print_iter_train),optimizer.lr))
                    print_traincorrects=0.0
                    print_trainloss=0.0    
                #val上面的精度大于0.88，开始好好地查找最合适的模型
#                if i%print_iter_val==0 and epoch>0 and best_acc>0.45:
#                    print_valcorrects=0.0
#                    for i in (range(int(val_data.count))):  
#                        inputs_val, labels_val = val_data.next_batch()#迭代器  
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
                    
#                    if val_acc>best_acc:
#                        best_acc = val_acc
#                        best_model_wts = model.state_dict()
                    
            epoch_loss = float(running_loss) / (Iter*data.batch_size)
            epoch_acc = float(running_corrects) / (Iter*data.batch_size)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 如果val的acc大于之前最好的val_acc
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            ## 保存模型
            torch.save(model,'../save_models/TSnet_model_{0}.pth'.format(epoch+31))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights 最后循环完，加入最好的module
    model.load_state_dict(best_model_wts)
    
    return model

###############################################
use_gpu = torch.cuda.is_available()
#定义网络
model=TSnet(17)
#model=resnetv2sn18()
#model=torch.load('/home/zj/senetial/save_models/TSnet_model_30.pth')

if use_gpu and len(device_ids)>1:#多gpu训练
    model = model.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)
if use_gpu and len(device_ids)==1:#单gpu训练
    model = model.cuda()

#定义损失函数
criterion = nn.CrossEntropyLoss()
#定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
#定义优化策略
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,50)
#定义trian的数据和val的数据
trian_data = Generator(filepath='/home/zj/senetial/data',datatype='train',batch_size=16,split=0.1)
val_data = Generator(filepath='/home/zj/senetial/data',datatype='val',batch_size=16,split=0.1)

model_ft = train_model(model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           num_epochs=50)




##TEST
#model_best = torch.load('/home/zj/senetial/save_models/model_1.pth')
#test_data = generotor_batch(batch_size=16,data_set='test')
#inputs = Variable((torch.from_numpy(test_data)).float().permute(0, 3, 1, 2).cuda())
#pre = model_best(inputs)
#pre_hot=torch.FloatTensor(inputs.shape[0],17)
#pre_hot.zero_()
#f=open('','w')
#for i in range(pre.shape[0]):
#    pre_hot[i,pre[i]]=1
#    f.write(pre_hot[i,:]+'\n')






















op
