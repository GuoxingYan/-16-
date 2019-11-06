# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
from torchsummary import summary


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class TSnet(nn.Module):

    def __init__(self, num_classes=17):
        self.inplanes = 128
        
        super(TSnet, self).__init__()
      
        self.layer1=nn.Sequential(conv3x3(18,self.inplanes),
                                  nn.ReLU(inplace=True),
                                  conv3x3(self.inplanes,self.inplanes),
                                  nn.ReLU(inplace=True),
                                  #nn.Dropout(p=0.1),
                                  conv3x3(self.inplanes,self.inplanes,2),
                                   )
        self.layer2=nn.Sequential(conv3x3(self.inplanes,self.inplanes*2),
                                  nn.ReLU(inplace=True),
                                  #nn.Dropout(p=0.1),
                                  conv3x3(self.inplanes*2,self.inplanes*2),
                                  nn.ReLU(inplace=True),
                                  #nn.Dropout(p=0.1),
                                  conv3x3(self.inplanes*2,self.inplanes*2,2),
                                   )
        self.layer3=nn.Sequential(conv3x3(self.inplanes*2,self.inplanes*4),
                                  nn.ReLU(inplace=True),
                                  #nn.Dropout(p=0.1),
                                  conv3x3(self.inplanes*4,self.inplanes*4),
                                  nn.ReLU(inplace=True),
                                  #nn.Dropout(p=0.1),
                                  nn.ReLU(inplace=True),
                                  conv3x3(self.inplanes*4,self.inplanes*4,2),
                                   )        
        self.fc=nn.Sequential(
             nn.Linear(4096*2,512),
             nn.Dropout(p=0.5),
             nn.LeakyReLU(inplace=True),#relu在drop前面还是后面
             nn.Linear(512,17))        
        self.softmax=nn.Softmax()

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        #x = nn.Dropout(p=0.5)
        x = self.fc(x)
        x=self.softmax(x)
        ####

        return x


if __name__=="__main__":
    tenet=TSnet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=tenet.to(device)
    summary(model,(18,32,32))
    
