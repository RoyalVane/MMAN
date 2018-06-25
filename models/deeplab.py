import torch.nn as nn
import torch
import numpy as np
from torchvision import models

affine_par = True

def get_bn_lr_params(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.bn1.parameters())
    b.append(model.layer1[0].downsample[1].parameters())
    for i in range(3):
        b.append(model.layer1[i].bn1.parameters())
        b.append(model.layer1[i].bn2.parameters())
        b.append(model.layer1[i].bn3.parameters())
    b.append(model.layer2[0].downsample[1].parameters())
    for j in range(4):
        b.append(model.layer2[j].bn1.parameters())
        b.append(model.layer2[j].bn2.parameters())
        b.append(model.layer2[j].bn3.parameters())
    b.append(model.layer3[0].downsample[1].parameters())
    for k in range(23):
        b.append(model.layer3[k].bn1.parameters())
        b.append(model.layer3[k].bn2.parameters())
        b.append(model.layer3[k].bn3.parameters())
    b.append(model.layer4[0].downsample[1].parameters())
    for m in range(3):
        b.append(model.layer4[m].bn1.parameters())
        b.append(model.layer4[m].bn2.parameters())
        b.append(model.layer4[m].bn3.parameters())

    for ii in range(len(b)):
        for jj in b[ii]:
            yield jj

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []
    b.append(model.conv1.parameters())
    b.append(model.layer1[0].downsample[0].parameters())
    for i in range(3):
        b.append(model.layer1[i].conv1.parameters())
        b.append(model.layer1[i].conv2.parameters())
        b.append(model.layer1[i].conv3.parameters())
    b.append(model.layer2[0].downsample[0].parameters())
    for j in range(4):
        b.append(model.layer2[j].conv1.parameters())
        b.append(model.layer2[j].conv2.parameters())
        b.append(model.layer2[j].conv3.parameters())
    b.append(model.layer3[0].downsample[0].parameters())
    for k in range(23):
        b.append(model.layer3[k].conv1.parameters())
        b.append(model.layer3[k].conv2.parameters())
        b.append(model.layer3[k].conv3.parameters())
    b.append(model.layer4[0].downsample[0].parameters())
    for m in range(3):
        b.append(model.layer4[m].conv1.parameters())
        b.append(model.layer4[m].conv2.parameters())
        b.append(model.layer4[m].conv3.parameters())

    for ii in range(len(b)):
        for jj in b[ii]:
            yield jj

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer5.parameters())

    for ii in range(len(b)):
        for jj in b[ii]:
            yield jj

class D_Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(D_Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv1_1 = nn.Conv2d(num_classes * 4, num_classes, kernel_size = 1)
        self.conv1_1.weight.data.normal_(0, 0.01)
        conv1 = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.conv2d_list.append(conv1)
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out = torch.cat([out, self.conv2d_list[i+1](x)], 1)
        out = self.conv1_1(out)
        return out

class D_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(D_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class D_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, input_size):
        self.inplanes = 64
        super(D_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # change
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation = [1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation = [1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation = [1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=[1,1,1], rate = 2)
        self.layer5 = self._make_pred_layer(D_Classifier_Module, [2,4,6],[2,4,6], num_classes)
        self.upsample = nn.Upsample(input_size, mode='bilinear')
        

    def _make_layer(self, block, planes, blocks, stride=1, dilation=[1], rate = 1):
        downsample = None
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=rate * dilation[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=rate * dilation[i] if len(dilation) > 1 else dilation[0]))
        return nn.Sequential(*layers)
    
    def _make_pred_layer(self,block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):
        sm = nn.Softmax2d()
        lsm = nn.LogSoftmax()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.upsample(x)
        return {'GAN':x, 'L1':lsm(x)}   