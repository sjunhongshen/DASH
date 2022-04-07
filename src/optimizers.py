import torch
from torch import nn, optim

class ExpGrad(optim.Optimizer):

    def __init__(self, params, lr):

        params = list(params)
        for param in params:
            if param.sum() - 1.0 > 0.01:
                param /= param.sum()
                # raise(ValueError("parameters must lie on the simplex"))
        super(ExpGrad, self).__init__(params, {'lr': lr})

     def step(self, closure=None):

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                p.data *= torch.exp(-lr * p.grad)
                p.data /= p.data.sum()
