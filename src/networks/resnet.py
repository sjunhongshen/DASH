'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from torch.autograd import Variable

__all__ = ['ResNet', 'ResNet_Sep', 'resnet20', 'resnet20_sep', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A',ks=None, ds=None):
        super(BasicBlock, self).__init__()
        k, d = 3 if ks is None else ks[0], 1 if ds is None else ds[0]
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=k, stride=stride, padding=k//2 * d, bias=False, dilation=d)
        self.bn1 = nn.BatchNorm2d(planes)
        k, d = 3 if ks is None else ks[1], 1 if ds is None else ds[1]
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=k, stride=1, padding=k//2 * d, bias=False, dilation=d)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::stride, ::stride], (0, 0, 0, 0, (planes - in_planes)//2, (planes - in_planes)//2), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, num_classes=10, ks=None, ds=None, strides=None, pool_k=1, remain_shape=False, squeeze=False, activation=None):
        super(ResNet, self).__init__()
        self.remain_shape = remain_shape
        self.squeeze = squeeze
        self.activation = activation
        self.pool_k = pool_k

        if self.remain_shape:
            strides = [1] * 4
        self.in_planes = 16
        if ks is None: ks = [3] * (2 * int(np.sum(num_blocks)) + 1)
        if ds is None: ds = [1] * (2 * int(np.sum(num_blocks)) + 1)
        k, d = ks[0], ds[0]
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=k, stride=1 if strides is None else strides[0], padding=k//2 * d, bias=False, dilation=d)
        self.bn1 = nn.BatchNorm2d(16)
        ks, ds = ks[1:], ds[1:]
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1 if strides is None else strides[1], ks=None if ks is None else ks[:2 * num_blocks[0]],ds=None if ds is None else ds[:2 * num_blocks[0]])
        ks, ds = ks[2 * num_blocks[0]:], ds[2 * num_blocks[0]:]
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2 if strides is None else strides[2], ks=None if ks is None else ks[:2 * num_blocks[1]],ds=None if ds is None else ds[:2 * num_blocks[1]])
        ks, ds = ks[2 * num_blocks[1]:], ds[2 * num_blocks[1]:]
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2 if strides is None else strides[3], ks=None if ks is None else ks[:2 * num_blocks[2]],ds=None if ds is None else ds[:2 * num_blocks[2]])
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, ks=None, ds=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, ks=ks[:2], ds=ds[:2]))
            self.in_planes = planes * block.expansion
            ks, ds = ks[2:], ds[2:]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if not self.remain_shape:
            if self.pool_k > 0:
                x = F.avg_pool2d(x, x.size()[3])
            else:
                x = x.view(x.size(0), x.size(1), -1).mean(-1)
            x = x.view(x.size(0), -1)
        else:
            x = x.permute(0, 2, 3, 1)
        
        x = self.linear(x)
        
        if self.remain_shape:
            x = x.permute(0, 3, 1, 2)
        if self.squeeze:
            x = x.squeeze()
        if self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x

class ResNet_Sep(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, num_classes=10, ks=None, ds=None, strides=None, pool_k=1, remain_shape=False, squeeze=False, activation=None):
        super(ResNet_Sep, self).__init__()
        self.remain_shape = remain_shape
        self.squeeze = squeeze
        self.activation = activation
        self.pool_k = pool_k

        if self.remain_shape:
            strides = [1] * 4
        self.in_planes = 16
        if ks is None: ks = [3] * (2 * int(np.sum(num_blocks)) + 1)
        if ds is None: ds = [1] * (2 * int(np.sum(num_blocks)) + 1)
        k, d = ks[0], ds[0]
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=k, stride=1 if strides is None else strides[0], padding=k//2 * d, bias=False, dilation=d)
        self.bn1 = nn.BatchNorm2d(16)
        ks, ds = ks[1:], ds[1:]
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1 if strides is None else strides[1], ks=None if ks is None else ks[:2 * num_blocks[0]],ds=None if ds is None else ds[:2 * num_blocks[0]])
        ks, ds = ks[2 * num_blocks[0]:], ds[2 * num_blocks[0]:]
        self.channel_match1 = nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=True)
        self.in_planes = 32

        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2 if strides is None else strides[2], ks=None if ks is None else ks[:2 * num_blocks[1]],ds=None if ds is None else ds[:2 * num_blocks[1]])
        ks, ds = ks[2 * num_blocks[1]:], ds[2 * num_blocks[1]:]
        self.channel_match2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=True)
        self.in_planes = 64

        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2 if strides is None else strides[3], ks=None if ks is None else ks[:2 * num_blocks[2]],ds=None if ds is None else ds[:2 * num_blocks[2]])
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, ks=None, ds=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, ks=ks[:2], ds=ds[:2]))
            self.in_planes = planes * block.expansion
            ks, ds = ks[2:], ds[2:]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.channel_match1(self.layer1(x))
        x = self.channel_match2(self.layer2(x))
        x = self.layer3(x)
        if not self.remain_shape:
            if self.pool_k > 0:
                x = F.avg_pool2d(x, x.size()[3])
            else:
                x = x.view(x.size(0), x.size(1), -1).mean(-1)
            x = x.view(x.size(0), -1)
        else:
            x = x.permute(0, 2, 3, 1)
        
        x = self.linear(x)
        
        if self.remain_shape:
            x = x.permute(0, 3, 1, 2)
        if self.squeeze:
            x = x.squeeze()
        if self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x


def resnet20(in_channel=3, num_classes=10, ks=None, ds=None, strides=None, pool_k=1, remain_shape=False, squeeze=False, activation=None):
    return ResNet(BasicBlock, [3, 3, 3], in_channel=in_channel, num_classes=num_classes, ks=ks, ds=ds, strides=strides, pool_k=pool_k, remain_shape=remain_shape, squeeze=squeeze, activation=activation)

def resnet20_sep(in_channel=3, num_classes=10, ks=None, ds=None, strides=None, pool_k=1, remain_shape=False, squeeze=False, activation=None):
    return ResNet_Sep(BasicBlock, [3, 3, 3], in_channel=in_channel, num_classes=num_classes, ks=ks, ds=ds, strides=strides, pool_k=pool_k, remain_shape=remain_shape, squeeze=squeeze, activation=activation)


def resnet32(in_channel=3, num_classes=10, ks=None, ds=None, strides=None, pool_k=1, remain_shape=False, squeeze=False, activation=None):
    return ResNet(BasicBlock, [5, 5, 5], in_channel=in_channel, num_classes=num_classes, ks=ks, ds=ds, strides=strides, pool_k=pool_k, remain_shape=remain_shape, squeeze=squeeze, activation=activation)


def resnet44(in_channel=3, num_classes=10, ks=None, ds=None, strides=None, pool_k=1, remain_shape=False, squeeze=False, activation=None):
    return ResNet(BasicBlock, [7, 7, 7], in_channel=in_channel, num_classes=num_classes, ks=ks, ds=ds, strides=strides, pool_k=pool_k, remain_shape=remain_shape, squeeze=squeeze, activation=activation)


def resnet56(in_channel=3, num_classes=10, ks=None, ds=None, strides=None, pool_k=1, remain_shape=False, squeeze=False, activation=None):
    return ResNet(BasicBlock, [9, 9, 9], in_channel=in_channel, num_classes=num_classes, ks=ks, ds=ds, strides=strides, pool_k=pool_k, remain_shape=remain_shape, squeeze=squeeze, activation=activation)


def resnet110(in_channel=3, num_classes=10, ks=None, ds=None, strides=None, pool_k=1, remain_shape=False, squeeze=False, activation=None):
    return ResNet(BasicBlock, [18, 18, 18], in_channel=in_channel, num_classes=num_classes, ks=ks, ds=ds, strides=strides, pool_k=pool_k, remain_shape=remain_shape, squeeze=squeeze, activation=activation)


def resnet1202(in_channel=3, num_classes=10, ks=None, ds=None, strides=None, pool_k=1, remain_shape=False, squeeze=False, activation=None):
    return ResNet(BasicBlock, [200, 200, 200], in_channel=in_channel, num_classes=num_classes, ks=ks, ds=ds, strides=strides, pool_k=pool_k, remain_shape=remain_shape, squeeze=squeeze, activation=activation)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))



if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
