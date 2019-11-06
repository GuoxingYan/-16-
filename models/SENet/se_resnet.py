import math
import torch
import torch.nn as nn
from torchvision.models import ResNet
from .se_module import SELayer
from .switchable_norm import *
# import switchable_norm as sn
from torchsummary import summary

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.sn1 = SwitchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.sn2 = SwitchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.sn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.sn1 = SwitchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.sn2 = SwitchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.sn3 = SwitchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.sn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.sn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet101(num_classes):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class LczSEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(LczSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.sn1 = SwitchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.sn2 = SwitchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            #self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            #                                sn(planes))
            #ResNet-D 
            self.downsample = nn.Sequential(nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
                                            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                                            SwitchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.sn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out
####       
class LczSEBasicBlockV2(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(LczSEBasicBlockV2, self).__init__()
        self.sn1 = SwitchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.sn2 = SwitchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)  
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
#            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
#                                            SwitchNorm2d(planes))
            #ResNet-D 
            self.downsample = nn.Sequential(nn.AvgPool2d(kernel_size=3,stride=2,padding=1),
                                            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                                            SwitchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)    
        out = self.sn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.sn2(out)
        out = self.relu(out)
        out = self.conv2(out)   
        out = self.se(out)
        out += residual

        return out



class LczSEResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=17, reduction=16):
        super(LczSEResNet, self).__init__()
        self.inplane = 128
        self.conv1 = nn.Conv2d(21, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.sn1 = SwitchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        ###
        #self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        #self.sn1 = SwitchNorm2d(self.inplane)
        #self.relu = nn.ReLU(inplace=True)
        ###
        self.layer1 = self._make_layer(block, 128, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(block, 256, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 512, blocks=n_size, stride=2, reduction=reduction)
        ###
        #self.layer4 = self._make_layer(block, 1024, blocks=1, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
        self.Softmax=nn.LogSoftmax()
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SwitchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, std=1e-3)


    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x=self.Softmax(x)

        return x


class LczSEPreActResNet(LczSEResNet):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(LczSEPreActResNet, self).__init__(block, n_size, num_classes, reduction)
        self.sn1 = SwitchNorm2d(self.inplane)
        self.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.sn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


def se_resnet20(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = LczSEResNet(LczSEBasicBlock, 2, **kwargs)
    return model
def se_resnet20_v2(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = LczSEResNet(LczSEBasicBlockV2, 3, **kwargs)
    return model



def se_resnet32(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = LczSEResNet(LczSEBasicBlock, 5, **kwargs)
    return model


def se_resnet56(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = LczSEResNet(LczSEBasicBlock, 9, **kwargs)
    return model


def se_preactresnet20(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = LczSEPreActResNet(LczSEBasicBlock, 3, **kwargs)
    return model


def se_preactresnet32(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = LczSEPreActResNet(LczSEBasicBlock, 5, **kwargs)
    return model


def se_preactresnet56(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = LczSEPreActResNet(LczSEBasicBlock, 9, **kwargs)
    return model

if __name__=="__main__":
    device = torch.device('cuda')
    model=se_resnet20_v2().to(device)
    summary(model,(21,32,32))
