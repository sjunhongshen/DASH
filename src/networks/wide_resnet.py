import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, k=3, d=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=k, dilation=d, stride=stride, padding=k//2 * d, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, ks=None, ds=None):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        k, d = 3 if ks is None else ks[0], 1 if ds is None else ds[0]
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=k, dilation=d, padding=k//2 * d, bias=True)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        k, d = 3 if ks is None else ks[1], 1 if ds is None else ds[1]
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=k, dilation=d, padding=k//2 * d, stride=stride, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True))

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, in_channel=3, pool_k = 8, strides=None, ks=None, ds=None, remain_shape=False, squeeze=False, activation=None):
        super(Wide_ResNet, self).__init__()
        self.remain_shape = remain_shape
        self.squeeze = squeeze
        self.activation = activation
        if self.remain_shape:
            strides = [1] * 4

        self.in_planes = 16
        self.pool_k = pool_k

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = int(widen_factor)

        # print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        if ks is None: ks = [3] * (6 * n + 1)
        if ds is None: ds = [1] * (6 * n + 1)
        k, d = ks[0], ds[0]
        self.conv1 = conv3x3(in_channel, nStages[0], k=k, d=d, stride=1 if strides is None else strides[0])
        ks, ds = ks[1:], ds[1:]
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1 if strides is None else strides[1], ks=None if ks is None else ks[:2 * n],ds=None if ds is None else ds[:2 * n])
        ks, ds = ks[2*n:], ds[2*n:]
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2 if strides is None else strides[2], ks=None if ks is None else ks[:2 * n],ds=None if ds is None else ds[:2 * n])
        ks, ds = ks[2*n:], ds[2*n:]
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2 if strides is None else strides[3], ks=None if ks is None else ks[:2 * n],ds=None if ds is None else ds[:2 * n])
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        self.apply(conv_init)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, ks=None, ds=None):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, ks=ks[:2], ds=ds[:2]))
            self.in_planes = planes
            ks, ds = ks[2:], ds[2:]

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.relu(self.bn1(out))
        if not self.remain_shape:
            if self.pool_k > 0:
                out = F.avg_pool2d(out, out.size(-1))
            else:
                out = out.view(out.size(0), out.size(1), -1).mean(-1)
            out = out.view(out.size(0), -1)
        else:
            out = out.permute(0, 2, 3, 1)

        out = self.linear(out)
        
        if self.remain_shape:
            out = out.permute(0, 3, 1, 2)
        if self.squeeze:
            out = out.squeeze()
        if self.activation == 'sigmoid':
            out = torch.sigmoid(out)
        return out

####################################################################################
class Wide_ResNet_Sep(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, in_channel=3, pool_k=8, strides=None, ks=None, ds=None, remain_shape=False, squeeze=False, activation=None):
        super(Wide_ResNet_Sep, self).__init__()
        self.remain_shape = remain_shape
        self.squeeze = squeeze
        self.activation = activation
        if self.remain_shape:
            strides = [1] * 4

        self.in_planes = 16
        self.pool_k = pool_k

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = int(widen_factor)

        # print('| Wide-Resnet-Sep %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        if ks is None: ks = [3] * (6 * n + 1)
        if ds is None: ds = [1] * (6 * n + 1)
        k, d = ks[0], ds[0]
        self.conv1 = conv3x3(in_channel, nStages[0], k=k, d=d, stride=1 if strides is None else strides[0])

        self.channel_match1 = nn.Conv2d(nStages[0], nStages[1], kernel_size=1, stride=1, bias=True)
        self.in_planes = nStages[1]

        ks, ds = ks[1:], ds[1:]
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1 if strides is None else strides[1], ks=None if ks is None else ks[:2 * n],ds=None if ds is None else ds[:2 * n])
        self.channel_match2 = nn.Conv2d(nStages[1], nStages[2], kernel_size=1, stride=1, bias=True)
        self.in_planes = nStages[2]

        ks, ds = ks[2*n:], ds[2*n:]
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2 if strides is None else strides[2], ks=None if ks is None else ks[:2 * n],ds=None if ds is None else ds[:2 * n])
        self.channel_match3 = nn.Conv2d(nStages[2], nStages[3], kernel_size=1, stride=1, bias=True)
        self.in_planes = nStages[3]

        ks, ds = ks[2*n:], ds[2*n:]
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2 if strides is None else strides[3], ks=None if ks is None else ks[:2 * n],ds=None if ds is None else ds[:2 * n])
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        self.apply(conv_init)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, ks=None, ds=None):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, ks=ks[:2], ds=ds[:2]))
            self.in_planes = planes
            ks, ds = ks[2:], ds[2:]

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.channel_match1(self.conv1(x))
        out = self.channel_match2(self.layer1(out))
        out = self.channel_match3(self.layer2(out))
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        if not self.remain_shape:
            if self.pool_k > 0:
                out = F.avg_pool2d(out, self.pool_k)
            else:
                out = out.view(out.size(0), out.size(1), -1).mean(-1)
            out = out.view(out.size(0), -1)
        else:
            out = out.permute(0, 2, 3, 1)
        
        out = self.linear(out)
        
        if self.remain_shape:
            out = out.permute(0, 3, 1, 2)
        if self.squeeze:
            out = out.squeeze()
        if self.activation == 'sigmoid':
            out = torch.sigmoid(out)
        return out
        
if __name__ == '__main__':
    net=Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())
