# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:33:44 2018

@author: ygx
"""
#自定义的模型resnet模型
import torch
from torchsummary import summary
from torch import nn

from torch.nn import functional as F

from torchvision import models

class ResidualBlock1(nn.Module):
    #实现子module: Residual    Block
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock1,self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        
        self.right=shortcut
        
    def forward(self,x):
        out=self.left(x)
        residual=x if self.right is None else self.right(x)
        out+=residual
        return F.relu(out)
        
class ResidualBlock2(nn.Module):
    #实现子module: Residual    Block
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock2,self).__init__()
        self.left=nn.Sequential(
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
        )
        
        self.right=shortcut
        
    def forward(self,x):
        out=self.left(x)
        residual=x if self.right is None else self.right(x)
        out+=residual
        return out
   
    
class ResNet(nn.Module):
    #实现主module:ResNet34
    #ResNet34包含多个layer,每个layer又包含多个residual block
    #用子module实现residual block , 用 _make_layer 函数实现layer
    def __init__(self,num_classes=17):
        super(ResNet,self).__init__()
        self.pre=nn.Sequential(
            nn.Conv2d(18,64,3,1,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(16)
        )###只有一个卷积，少了一个
        #重复的layer,分别有3,4,6,3个residual block
        self.layer1=self._make_layer(64,64,3)
        self.layer2=self._make_layer(64,128,4,stride=2)
        self.layer3=self._make_layer(128,256,6,stride=2)
        #self.layer4=self._make_layer(256,512,3,stride=2)
        #nn.Linear(512,num_classes)
        #分类用的全连接
        self.fc=nn.Sequential(
             nn.Linear(256,1000),
             nn.LeakyReLU(inplace=True),
             nn.Linear(1000,256),
             nn.Dropout(p=0.8),
             nn.LeakyReLU(inplace=True),#relu在drop前面还是后面
             nn.Linear(256,17))
            
        
        
    def _make_layer(self,inchannel,outchannel,block_num,stride=1):
        #构建layer,包含多个residual block
        shortcut=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel))

        layers=[ ]
        layers.append(ResidualBlock2(inchannel,outchannel,stride,shortcut))
        
        for i in range(1,block_num):
            layers.append(ResidualBlock2(outchannel,outchannel))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.pre(x)
        
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        #x=self.layer4(x)        
        x=F.max_pool2d(x,4)
        x=x.view(x.size(0),-1)
        
        return self.fc(x)

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ResNet().to(device)
    summary(model,(18,32,32))
    


