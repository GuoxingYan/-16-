# -*- coding: utf-8 -*-
"""
这个是tsnetSE+labelsmooth结构
"""
import sys
import torch.nn as nn
import torch
from torchsummary import summary
import switchable_norm as sn
import math
sys.path.append('/home/zj/senetial/models')
from SENet.se_module import SELayer



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class TSnetSE(nn.Module):

    def __init__(self, num_classes=17):
        self.inplanes = 128
        
        super(TSnetSE, self).__init__()
      
        self.layer1=nn.Sequential(conv3x3(21,self.inplanes),
                                  sn.SwitchNorm2d(self.inplanes),
                                  nn.ReLU(inplace=True),
                                  conv3x3(self.inplanes,self.inplanes),
                                  sn.SwitchNorm2d(self.inplanes),
                                  nn.ReLU(inplace=True),
                                  conv3x3(self.inplanes,self.inplanes,2),
                                  sn.SwitchNorm2d(self.inplanes),
                                  nn.ReLU(inplace=True),
                                   )
        self.se1=SELayer(self.inplanes)
        self.layer2=nn.Sequential(conv3x3(self.inplanes,self.inplanes*2),
                                  sn.SwitchNorm2d(self.inplanes*2),
                                  nn.ReLU(inplace=True),
                                  conv3x3(self.inplanes*2,self.inplanes*2),
                                  sn.SwitchNorm2d(self.inplanes*2),
                                  nn.ReLU(inplace=True),
                                  conv3x3(self.inplanes*2,self.inplanes*2,2),
                                  sn.SwitchNorm2d(self.inplanes*2),
                                  nn.ReLU(inplace=True),
                                   )
        self.se2=SELayer(self.inplanes*2)
        self.layer3=nn.Sequential(conv3x3(self.inplanes*2,self.inplanes*4),
                                  sn.SwitchNorm2d(self.inplanes*4),
                                  nn.ReLU(inplace=True),
                                  conv3x3(self.inplanes*4,self.inplanes*4),
                                  sn.SwitchNorm2d(self.inplanes*4),
                                  nn.ReLU(inplace=True),
                             
                                  conv3x3(self.inplanes*4,self.inplanes*4,2),
                                  sn.SwitchNorm2d(self.inplanes*4),
                                   ) 
        self.se3=SELayer(self.inplanes*4)
        self.fc=nn.Sequential(
             nn.Linear(4096*2,1024),
             nn.LeakyReLU(inplace=True),
             nn.Dropout(p=0.5),
             nn.Linear(1024,512),
             nn.LeakyReLU(inplace=True),#relu在drop前面还是后面
             nn.Dropout(p=0.8),
             nn.Linear(512,17))        
        self.LogSoftmax=nn.LogSoftmax()
        #后添加
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight)
                #bag of tricks
                n = m.in_channels + m.out_channels
                m.weight.data.normal_(-math.sqrt(6. / n), math.sqrt(6. / n))
            elif isinstance(m, sn.SwitchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x=self.LogSoftmax(x)
        ####
        return x


if __name__=="__main__":
    tenet=TSnetSE()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=tenet.to(device)
    summary(model,(21,32,32))
    
