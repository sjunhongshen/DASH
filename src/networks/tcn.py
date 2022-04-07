import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.nn.init as init

def bn_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.remove = False

    def forward(self, x):
        if self.remove:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2, out_size=784, bn=True):
        super(TemporalBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size[0], stride=stride, padding=(kernel_size[0] - 1) * dilation[0], dilation=dilation[0]), dim=None)
        self.chomp1 = Chomp1d((kernel_size[0] - 1) * dilation[0])
        self.dropout1 = nn.Dropout(dropout)

        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size[1], stride=stride, padding=(kernel_size[1] - 1) * dilation[1], dilation=dilation[1]), dim=None)
        self.chomp2 = Chomp1d((kernel_size[1] - 1) * dilation[1])
        self.dropout2 = nn.Dropout(dropout)

        if bn:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu, self.dropout1,
                                     self.conv2, self.chomp2, self.bn2, self.relu, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu, self.dropout1,
                                     self.conv2, self.chomp2, self.relu, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dilate=True, out_size=784, ks=None, ds=None, bn=True):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            kernel_size_ = [kernel_size, kernel_size] if ks is None else ks[2 * i: 2 * i + 2]
            dilation_size_ = [2 ** i, 2 ** i] if dilate else [1, 1]
            dilation_size_ = dilation_size_ if ds is None else ds[2 * i: 2 * i + 2]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size=kernel_size_, stride=1, dilation=dilation_size_, dropout=dropout, out_size=out_size, bn=bn)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, activation='softmax', remain_shape=False, bn=True, **kwargs):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, bn=bn, **kwargs)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.activation = activation
        self.remain_shape = remain_shape
        self.bn = nn.BatchNorm1d(num_channels[-1]) if bn else None

        self.apply(bn_init)

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = self.tcn(x)  # input should have dimension (N, C, L)
        if self.bn is not None:
            x = self.bn(x)
        x = x.permute(0, 2, 1) if self.remain_shape else x[:, :, -1]
        x = self.linear(x)
        if self.remain_shape:
            x = x.permute(0, 2, 1)
        if self.activation == 'softmax':
            return F.log_softmax(x, dim=1)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        else:
            return x