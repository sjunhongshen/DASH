import torch
import torch.nn as nn
import torch.nn.functional as F
from relax.ops import FNO

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2,  width, op='fno', ks=None, ds=None, einsum=True, padding_mode='circular'):
        super(SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        if op == 'fno':
            self.conv0 = FNO(self.width, self.width,
                [self.modes1, self.modes2], pad=True, einsum=einsum)
            self.conv1 = FNO(self.width, self.width,
                [self.modes1, self.modes2], pad=True, einsum=einsum)
            self.conv2 = FNO(self.width, self.width,
                [self.modes1, self.modes2], pad=True, einsum=einsum)
            self.conv3 = FNO(self.width, self.width,
                [self.modes1, self.modes2], pad=True, einsum=einsum)
        elif op == 'conv':
            if ks is not None:
                self.conv0 = nn.Conv2d(self.width, self.width,
                    kernel_size=ks[0], dilation=ds[0], padding=ks[0]//2 * ds[0],
                    bias=False)
                self.conv1 = nn.Conv2d(self.width, self.width,
                    kernel_size=ks[1], dilation=ds[1], padding=ks[1]//2 * ds[1],
                    bias=False)
                self.conv2 = nn.Conv2d(self.width, self.width,
                    kernel_size=ks[2], dilation=ds[2], padding=ks[2]//2 * ds[2],
                    bias=False)
                self.conv3 = nn.Conv2d(self.width, self.width,
                    kernel_size=ks[3], dilation=ds[3], padding=ks[3]//2 * ds[3],
                    bias=False)
            else:
                self.conv0 = nn.Conv2d(self.width, self.width,
                    kernel_size=self.modes1 + 1, padding=6,
                    padding_mode=padding_mode, bias=False)
                self.conv1 = nn.Conv2d(self.width, self.width,
                    kernel_size=self.modes1 + 1, padding=6,
                    padding_mode=padding_mode, bias=False)
                self.conv2 = nn.Conv2d(self.width, self.width,
                    kernel_size=self.modes1 + 1, padding=6,
                    padding_mode=padding_mode, bias=False)
                self.conv3 = nn.Conv2d(self.width, self.width,
                    kernel_size=self.modes1 + 1, padding=6,
                    padding_mode=padding_mode, bias=False)
        else:
            raise NotImplementedError

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Net2d(nn.Module):
    def __init__(self, modes, width, op='fno', ks=None, ds=None, **kwargs):
        super(Net2d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock2d(modes, modes, width, op, ks, ds, **kwargs)


    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c



#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


