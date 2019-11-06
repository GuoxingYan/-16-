# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:33:44 2018
原先愚蠢的认为都读到内存再打乱，可惜了h5文件太大了
@author: ygx
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


#定义路径
base_dir = os.path.expanduser('/home/zj/senetial/data')
path_train = os.path.join(base_dir,'training.h5')
path_val = os.path.join(base_dir,'validation.h5')
#打开h5文件
fid_train = h5py.File(path_train,'r')
fid_val = h5py.File(path_val,'r')
#查看数据存储形态
for key in fid_train.keys():
    print fid_train[key].name
    print fid_train[key].shape
    

    
s1_train = np.array(fid_train['sen1'],dtype=np.float16) #(352366, 32, 32, 8)
s2_train = np.array(fid_train['sen2'],dtype=np.float16) #(352366, 32, 32, 10)
label_train = np.array(fid_train['label'],dtype=np.int8)


s1_val = np.array(fid_val['sen1'],dtype=np.float16)
s2_val = np.array(fid_val['sen2'],dtype=np.float16)
label_val = np.array(fid_val['label'],dtype=np.int8)
fid_train.close()#关闭文件
fid_val.close()
#叠加到一起构成trianval
s1_trainval=np.vstack((s1_train,s1_val))
label_trainval=np.vstack((label_train,label_val))
s2_trainval=np.vstack((s2_train,s2_val))
print s1_trainval.shape,s2_trainval.shape,label_trainval.shape
#进行打乱操作
np.random.seed(0)
permulation = np.random.permutation(s1_trainval.shape[0])
shuffled_s1_trainval = s1_trainval[permulation,:,:,:]
shuffled_s2_trainval = s2_trainval[permulation,:,:,:]
shuffled_label_trainval = label_trainval[permulation,:]
#验证操作的正确性
#print '*'*60 #根据seed0得到的permulation得知下面两个应该相等
#print s1_trainval[0,:,:,:]==shuffled_s1_trainval[39,:,:,:]

#写出到新的.h文件当中
fid_train_new = h5py.File(os.path.join(base_dir,'training_new.h5'),'w')
fid_val_new = h5py.File(os.path.join(base_dir,'valiadtion_new.h5'),'w')
fid_train_new['sen1']=shuffled_s1_trainval[:352366,:,:,:]
fid_train_new['sen2']=shuffled_s2_trainval[:352366,:,:,:]
fid_train_new['label']=shuffled_label_trainval[:352366,:]
fid_val_new['sen1']=shuffled_s1_trainval[352366:,:,:,:]
fid_val_new['sen2']=shuffled_s2_trainval[352366:,:,:,:]
fid_val_new['label']=shuffled_label_trainval[352366:,:]
fid_train_new.close()
fid_val_new.close()


