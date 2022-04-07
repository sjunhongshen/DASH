import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
import operator

def logCoshLoss(y_t, y_prime_t, reduction='mean', eps=1e-12):
    if reduction == 'mean':
        reduce_fn = torch.mean
    elif reduction == 'sum':
        reduce_fn = torch.sum
    else:
        reduce_fn = lambda x: x
    x = y_prime_t - y_t
    return reduce_fn(torch.log((torch.exp(x) + torch.exp(-x)) / 2))

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)

class DeepCon(nn.Module):
    def __init__(self, L=64, num_blocks=8, width=16, expected_n_channels=57, no_dilation=False, ks=None, ds=None):
        super(DeepCon, self).__init__()
        self.L = L
        self.num_blocks = num_blocks
        self.width = width
        self.expected_n_channels = expected_n_channels

        self.downsample = nn.Sequential(
            nn.BatchNorm2d(expected_n_channels), 
            nn.ReLU(),
            nn.Conv2d(expected_n_channels, width, 
                kernel_size=(1, 1), stride=1))

        dropout_value = 0.3
        n_channels = width
        d_rate = 1
        dilation = not no_dilation
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            k = [3, 3] if ks is None else ks[2 * i: 2 * (i + 1)]
            d = [1, d_rate] if ds is None else ds[2 * i: 2 * (i + 1)]
            self.blocks.append(nn.Sequential(
                nn.BatchNorm2d(n_channels), 
                nn.ReLU(),
                nn.Conv2d(n_channels, n_channels, 
                    kernel_size=k[0], dilation = d[0], stride=1, padding=(k[0]-1)*d[0]//2),
                nn.Dropout2d(dropout_value),
                nn.ReLU(),
                nn.Conv2d(n_channels, n_channels, 
                    kernel_size=k[1], dilation = d[1], stride=1, padding=(k[1]-1)*d[1]//2)))
            if dilation:
                if d_rate == 1:
                    d_rate = 2
                elif d_rate == 2:
                    d_rate = 4
                else:
                    d_rate = 1

        k = 3 if ks is None else ks[-1]
        d = 1 if ds is None else ds[-1]
        self.out_layer = nn.Sequential(
            nn.BatchNorm2d(n_channels), 
            nn.ReLU(),
            nn.Conv2d(n_channels, 1, 
                kernel_size=k, dilation = d, stride=1, padding=(k-1)*d//2),
            nn.ReLU())

        self.apply(weight_init)

    def forward(self, x):
        tower = self.downsample(x)
        for block in self.blocks:
            b = block(tower)
            tower = b + tower
        output = self.out_layer(tower)
        return output

    def forward_window(self, x, stride=32):
        L = self.L
        _, _, _, s_length = x.shape

        if stride == -1: # Default to window size
            stride = L
            assert(s_length % L == 0)
        
        # Convert to numpy? 
        y = torch.zeros_like(x)[:, :1, :, :] # TODO Use nans? Use numpy?
        counts = torch.zeros_like(x)[:, :1, :, :]
        for i in range((((s_length - L) // stride)) + 1):
            ip = i * stride
            for j in range((((s_length - L) // stride)) + 1):
                jp = j * stride
                out = self.forward(x[:, :, ip:ip+L, jp:jp+L])
                y[:, :, ip:ip+L, jp:jp+L] += out
                counts[:, :, ip:ip+L, jp:jp+L] += torch.ones_like(out)

        return y / counts

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

