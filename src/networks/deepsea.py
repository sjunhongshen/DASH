import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSEA(nn.Module):
    def __init__(self, ks=None,ds=None):
        super(DeepSEA, self).__init__()
        k, d = 8 if ks is None else ks[0], 1 if ds is None else ds[0]
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=k, dilation=d, padding=k//2 * d)
        k, d = 8 if ks is None else ks[1], 1 if ds is None else ds[1]
        self.conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=k, dilation=d, padding=k//2 * d)
        k, d = 8 if ks is None else ks[2], 1 if ds is None else ds[2]
        self.conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=k, dilation=d, padding=k//2 * d)
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(59520, 925)
        self.linear2 = nn.Linear(925, 36)

    def forward(self, input):
        # print("input", input.shape)
        s = input.shape[-1]
        x = self.conv1(input)[..., :s]
        x = F.relu(x)
        # print("1", x.shape)
        x = self.maxpool(x)
        # print("2", x.shape)
        x = self.drop1(x)
        # print("3", x.shape)
        s = x.shape[-1]
        x = self.conv2(x)[..., :s]
        # print("4", x.shape)
        x = F.relu(x)
        x = self.maxpool(x)
        # print("5", x.shape)
        x = self.drop1(x)
        # print("6", x.shape)
        s = x.shape[-1]
        x = self.conv3(x)[..., :s]
        # print("7", x.shape)
        x = F.relu(x)
        x = self.drop2(x)
        # print("8", x.shape)
        x = x.view(x.size(0), -1)
        # print("9", x.shape)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
