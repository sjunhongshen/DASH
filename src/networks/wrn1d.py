''' modified code from https://github.com/okrasolar/pytorch-timeseries'''
import torch
from torch import nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.remove = False

    def forward(self, x):
        if self.remove:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, dilation: int, dropout_rate: float) -> None:
        super().__init__()
        self.chomp = Chomp1d((kernel_size - 1) * dilation)
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              dilation=dilation,
                              padding=(kernel_size - 1) * dilation,
                              stride=stride),
            self.chomp,
            nn.Dropout(p=dropout_rate),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)

class ResNet1D(nn.Module):
    """A PyTorch implementation of the ResNet Baseline
    From https://arxiv.org/abs/1909.04939
    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    mid_channels:
        The 3 residual blocks will have as output channels:
        [mid_channels, mid_channels * 2, mid_channels * 2]
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, mid_channels: int = 64,
                 num_pred_classes: int = 1,  dropout_rate: float = 0, activation=None, remain_shape=False, ks=None, ds=None) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(*[
            ResNetBlock(in_channels=in_channels, out_channels=mid_channels, dropout_rate=dropout_rate,
                ks=ks[:3] if ks is not None else None, ds=ds[:3] if ds is not None else None),
            ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2, dropout_rate=dropout_rate,
                ks=ks[3:6] if ks is not None else None, ds=ds[3:6] if ds is not None else None),
            ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2, dropout_rate=dropout_rate,
                ks=ks[6:] if ks is not None else None, ds=ds[6:] if ds is not None else None),

        ])
        self.final = nn.Linear(mid_channels * 2, num_pred_classes)
        self.activation = activation
        self.remain_shape = remain_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.layers(x)
        if self.remain_shape:
            x = x.permute(0, 2, 1)
        else:
            x = x.mean(dim=-1)
        x = self.final(x)
        if self.remain_shape:
            x = x.permute(0, 2, 1)
        if self.activation == 'softmax':
            return F.log_softmax(x, dim=1)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        else:
            return x


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float, ks=None, ds=None) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3] if ks is None else ks
        dilations = [1, 1, 1] if ds is None else ds

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i], stride=1, dilation = dilations[i], dropout_rate=dropout_rate) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)
