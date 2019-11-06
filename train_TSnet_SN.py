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
#可视化
#from tensorboardX import SummaryWriter
#writer = SummaryWriter('log')

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
#        np.random.seed(0)
#        self.permulation = np.random.permutation(self.count)  
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
#            if self.index == 0:
#                #跑完一个轮回，重新打乱
#                self.permulation = np.random.permutation(self.count)
#                self.index=1
            if self.data_set=='train' or self.data_set=='val':
                #self.image_s1[i,:,:,:]=np.expand_dims(np.array(self.s1[self.permulation[self.index-1],:,:,:]),axis=0)
                #self.image_s2[i,:,:,:]=np.expand_dims(np.array(self.s2[self.permulation[self.index-1],:,:,:]),axis=0)
#                self.labels[i,:]=np.expand_dims(np.array(self.label[self.permulation[self.index-1],:]),axis=0)        
           #不进行打乱操作
                self.image_s1[i,:,:,:]=np.expand_dims(np.array(self.s1[self.index-1,:,:,:]),axis=0)
                self.image_s2[i,:,:,:]=np.expand_dims(np.array(self.s2[self.index-1,:,:,:]),axis=0)            
                self.labels[i,:]=np.expand_dims(np.array(self.label[self.index-1,:]),axis=0)
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
#import time
#start=time.time()
#train = generotor_batch()
#计算方11011
#mean,std=0,0

#for i in range(1):
# 前30000个均值，方差
#   datas_train,_= train.__next__()#一次跑出batchsize张 32
#   mean =np.mean(datas_train,axis=(0,1,2))
#   std=np.std(datas_train,axis=(0,1,2))
#   print '_'*60
#   print mean
#   print '_'*60
#   print std 
#end=time.time()
#print end-start #100次1.14秒

#train_transform =transforms.Compose([transforms.Normalize([0.62,0.5,0.52,0.52,0,0,0.4,0.3,
#                                                           0.26,0.04,0.04,0.04,0.04,0.06,
#                                                           0.07,0.06,0.07,0.05,0.04],
#                                                          [0.001,0.0008,0.002,0.002,0.0003,0.0008,
#                                                           0.0003,0.0003,0.015,0.017,0.024,0.0228,
#                                                           0.027,0.032,0.033,0.036,0.0036,0.0316])])
##      

#train_transform =transforms.Compose([transforms.Resize(42),
#                                     transforms.RandomHorizontalFlip(0.5),
#                                     transforms.RandomVerticalFlip(0.5),
#                                     transforms.RandomCrop(32)])                                              

def train_model(model, criterion, optimizer, scheduler,batch_size=16, num_epochs=25):

    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        #开始第几次循环
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        data = trian_data
        scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        Iter=int(data.count/batch_size)
###输入前16000，验证下效果              
        #Iter=1000
        for i in tqdm(range(Iter)):#用1.6w张图片看下效果
              
            inputs, labels = data.next_batch()#迭代器  
            inputs = Variable((torch.from_numpy(inputs)).float().permute(0, 3, 1, 2).cuda())#输入必须是float N C H W
            #inputs = train_transform(inputs)
            labels = Variable((torch.from_numpy(labels)).long().cuda())#label必须是long
            optimizer.zero_grad()
            outputs = model(inputs)
            labels =labels.argmax(dim=1)#CE默认不支持one-hot编码
            _, preds = torch.max(outputs.data, 1)#这里已经转成了
            loss = criterion(outputs, labels)#CE默认不支持one-hot编码
     
            loss.backward()
            optimizer.step() 
            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)
            #每500个batch终端显示一次当前的loss和accu
            print_train=500
            if i%print_train==0:
                print('\r Loss: {:.4f} Acc: {:.4f}'.format(running_loss/print_train,running_corrects/print_train))                   
                running_loss=0.0
                running_corrects=0.0
            #每10个batch画个点用于loss曲线
            if i % 10 == 0:
                niter = epoch * (Iter) + i
                #writer.add_scalar('Train/Loss', loss.data[0], niter)
            #每2000个batch全验证集检测，画个点用于Accu
            print_val = 2000
            if i%print_val==0:
            	niter = epoch * (Iter) + i
            	correct = 0
            	total = 0
            	for j in range(int(val_data.count/batch_size)):
            		inputs,labels=val_data.next_batch()
            		inputs=Variable((torch.from_numpy(inputs)).float().permute(0, 3, 1, 2).cuda())
            		labels=Variable((torch.from_numpy(labels)).long().cuda()).argmax(dim=1)
              
                	res = model(inputs)
                	_, predicted = torch.max(res.data, 1)
                	total += labels.shape[0]
                	correct += torch.sum(predicted == labels.data)
           		print('Test_%4f/Accu_%4f', correct/total, niter)
           		#writer.add_scalar('Test/Accu', correct/total, niter)
           		epoch_acc =float(correct/total)
           		if epoch_acc > best_acc and epoch>0:
           			best_acc = epoch_acc
           			best_model_wts = model.state_dict()
           			print 'best_model_%d',epoch               

                best_acc = epoch_acc
                best_model_wts = model.state_dict() 
                              
            torch.save(model,'../save_models/TSnet_model_{0}.pth'.format(epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

###############################################
use_gpu = torch.cuda.is_available()
#定义网络
model=TSnet(17)
#model=resnetv2sn18()
#model=torch.load('../save_models/model_1.pth')

if use_gpu and len(device_ids)>1:#多gpu训练
    model = model.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)
if use_gpu and len(device_ids)==1:#单gpu训练
    model = model.cuda()
print model
#定义损失函数
#criterion =FocalLoss(17)
criterion = nn.CrossEntropyLoss()
#定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=0.001)
#定义优化策略
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
#定义trian的数据和val的数据
batch_size=16
trian_data = generotor_batch('/home/zj/senetial/data',batch_size,'train')
val_data = generotor_batch('/home/zj/senetial/data',batch_size,data_set='val')

model_ft = train_model(model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           batch_size=16,
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






















