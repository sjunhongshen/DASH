import os
import json
import argparse
import time
import numpy as np
import random
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from operator import itemgetter

from relax.nas import MixedOptimizer, Supernet, get_module
from dash import MixtureSupernet, elementwise_max, get_weight
from task_configs import get_data, get_config, get_model, get_metric
from task_utils import count_params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def time_search(model, loss, optimizer, scheduler, train_loader, transform, decoder, args, kernel_choices, dims, clip, share_param=False):
    for ep in range(2):
        
        if ep == 1:
            torch.cuda.synchronize()
            t = time.perf_counter()
        
        model.train()

        optimizer.zero_grad()

        for i, data in enumerate(train_loader):
            
            x, y = data 
            x, y = x.to(args.device), y.to(args.device)
            out = model(x)

            l = loss(out, y)
            l.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

    torch.cuda.synchronize()
    return time.perf_counter() - t


def main():

    parser = argparse.ArgumentParser(description='DASH')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--arch', type=str, default='', help='backbone architecture')
    parser.add_argument('--experiment_id', type=str, default='0', help='directory name to save the experiment results')
    parser.add_argument('--separable', type=int, default=1, help='use separable conv')
    parser.add_argument('--stream', type=int, default=1, help='use streaming for implementing aggconv')
    parser.add_argument('--test_input_size', type=int, default=0, 
        help='if 1, compare three methods by varying the input sizes; if 0, compare three methods by varying the kernel choices')

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tid = args.experiment_id
    args.save_dir = 'results_speed/'  + args.dataset + '/' + ('default' if len(args.arch) == 0 else args.arch) +'/' + tid
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(args.save_dir + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print(args.__dict__)

    torch.cuda.empty_cache()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0) 
    torch.cuda.manual_seed_all(0)

    dims, sample_shape, num_classes, batch_size, epochs, loss, lr, arch_lr, weight_decay, opt, arch_opt, weight_sched_search, weight_sched_train, accum, clip, retrain_clip, validation_freq, retrain_freq, \
    einsum, retrain_epochs, arch_default, kernel_choices_default, dilation_choices_default, quick_search, quick_retrain, config_kwargs = get_config(args.dataset)  
    
    batch_size = batch_size // 2
    
    arch = args.arch if len(args.arch) > 0 else arch_default
   
    kernel_choices_all = [3,5,7,9,11,13,15]
    dilation_choices_all = [1,3,7,15,31,63,127]

    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(args.dataset, batch_size, arch, False)
    train_loader, _ = load_mnist(batch_size, input_size=1000)
    sample_shape = (1, 1, 1000)
    
    decoder = data_kwargs['decoder'] if data_kwargs is not None and 'decoder' in data_kwargs else None 
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None 

    if args.device == 'cuda':
        try:
            loss.cuda()
        except:
            pass
        if decoder is not None:
            decoder.cuda()

    print("search arch:", arch, "\tbatch size:", batch_size, "\tnum train batch:", n_train)
    
    time_all = []
    config_all = []
    
    for i in range(1, len(kernel_choices_all) + 1):

        if args.test_input_size:
            kernel_choices = kernel_choices_all[:5]
            dilation_choices = dilation_choices_all[:5]
            train_loader, _ = load_mnist(batch_size, input_size=int(2 ** (4 + i)))
            sample_shape = (1, 1, int(2 ** (4 + i)))
        else:
            kernel_choices = kernel_choices_all[:i]
            dilation_choices = dilation_choices_all[:i]
            
        print('\nkernel choices:', kernel_choices, 'dilation choices:', dilation_choices, 'input size:', sample_shape[-1])

        times = {'dash': [], 'mixed-results': [], 'mixed-weights':[]}
        torch.cuda.empty_cache()
                
        for _ in range(3):
            model = get_model(arch, sample_shape, num_classes, config_kwargs)
            model = BaselineSupernet.create(model.cpu(), in_place=True)
            model.conv2baseline(kernel_sizes=kernel_choices, dilations=dilation_choices, stream=args.stream, dims=dims, separable=args.separable, relaxation=MixedWeights)
            model.remove_module("chomp")

            opts = [opt(model.parameters(), lr=lr, weight_decay=weight_decay)]
            optimizer = MixedOptimizer(opts)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=weight_sched_search)
            if args.device == 'cuda':
                model.cuda()
            times['mixed-weights'].append(time_search(model, loss, optimizer, scheduler, train_loader, transform, decoder, args, kernel_choices, dims, clip))

            print("mixed-weights", count_params(model), times['mixed-weights'][-1])
            del model

                
            model = get_model(arch, sample_shape, num_classes, config_kwargs)
            model = MixtureSupernet.create(model.cpu(), in_place=True)
            model.conv2mixture(torch.zeros(sample_shape),  kernel_sizes=kernel_choices, dilations=dilation_choices, dims=dims, separable=args.separable,
                    stream=args.stream, device=args.device, einsum=einsum, **config_kwargs)
            model.remove_module("chomp")

            opts = [opt(model.parameters(), lr=lr, weight_decay=weight_decay)]
            optimizer = MixedOptimizer(opts)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=weight_sched_search)
            if args.device == 'cuda':
                model.cuda()
                
            times['dash'].append(time_search(model, loss, optimizer, scheduler, train_loader, transform, decoder, args, kernel_choices, dims, clip))
            print("dash", count_params(model), times['dash'][-1])
            del model


            model = get_model(arch, sample_shape, num_classes, config_kwargs)
            model = BaselineSupernet.create(model.cpu(), in_place=True)
            model.conv2baseline(kernel_sizes=kernel_choices, dilations=dilation_choices, stream=args.stream, dims=dims, separable=args.separable)
            model.remove_module("chomp")

            opts = [opt(model.parameters(), lr=lr, weight_decay=weight_decay)]
            optimizer = MixedOptimizer(opts)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=weight_sched_search)
            if args.device == 'cuda':
                model.cuda()
            times['mixed-results'].append(time_search(model, loss, optimizer, scheduler, train_loader, transform, decoder, args, kernel_choices, dims, clip))
            print("mixed-results", count_params(model), times['mixed-results'][-1])
            del model
             

        for n, t in sorted(times.items(), key=itemgetter(0)):
            print(n, round(np.mean(t), 2), 'seconds per epoch', sep='\t')

        time_all.append(times)
        config_all.append(kernel_choices + dilation_choices)

        with open(os.path.join(args.save_dir, 'time_all.npy'), 'w') as f:
            json.dump(time_all, f)
        with open(os.path.join(args.save_dir, 'config_all.npy'), 'w') as f:
            json.dump(config_all, f)


"""Baselines"""

from relax.ops import Conv
from relax.xd import int2tuple, pad2size

class MixedResults(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, dilations=[1], stream=False, **kwargs):

        super(MixedResults, self).__init__()
        dims = len(kernel_sizes[0])
        self.kd = list(product(kernel_sizes, 
                               [int2tuple(d, length=dims) for d in dilations]))
        self.convs = nn.ModuleList(Conv(dims)(in_channels,
                                              out_channels,
                                              ks,
                                              dilation=ds,
                                              padding=[d*(k-1)//2 for k, d in zip(ks, ds)],
                                              **kwargs)
                                   for ks, ds in self.kd)
        self.alpha = nn.Parameter(torch.zeros(len(self.kd)))
        self.stream = stream
        self.temp = 1

    def forward(self, x):

        coef = get_weight(self.temp, self.alpha)

        if self.stream:
            for i, conv in enumerate(self.convs):
                if i == 0:
                    out = coef[i] * conv(x)
                else:
                    out += coef[i] * conv(x)
            return out
        for i, conv in enumerate(self.convs):
            out = conv(x)
            if i == 0:
                cat = torch.empty(*out.shape, len(self.kd), device=out.device)
            cat[...,i] = out
        return F.linear(cat, coef)


class MixedWeights(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, dilations=[1], stream=False, **kwargs):

        super(MixedWeights, self).__init__()
        dims = len(kernel_sizes[0])
        self.kd = list(product(kernel_sizes, 
                               [int2tuple(d, length=dims) for d in dilations]))
        self.weights = nn.ParameterList(nn.Parameter(torch.randn(out_channels, in_channels, *k)) for k, _ in self.kd)
        self.alpha = nn.Parameter(torch.zeros(len(self.kd)))
        self.shape = (out_channels, in_channels) + elementwise_max([[d*(k-1)+1 for k, d in zip(ks, ds)] for ks, ds in self.kd])
        self.kwargs = kwargs
        self.stream = stream
        self.dims = dims
        self.dim = len(kernel_sizes[0])
        self.temp = 1

    def forward(self, x):

        coef = get_weight(self.temp, self.alpha)
        if not self.stream:
            cat = torch.empty(*self.shape, len(self.kd), device=x.device)

        for i, ((ks, ds), weight) in enumerate(zip(self.kd, self.weights)):

            for j, (k, d) in enumerate(zip(ks, ds)):
                if d > 1:
                    p = torch.zeros((1, 1) + tuple([d] * self.dim), device=weight.device)
                    p[tuple([0] * (self.dim + 2))] = 1 #tuple([0] * self.dim)
                    if self.dims == 1:
                        weight = torch.kron(weight.transpose(-1, j+2).contiguous(),p)[...,:-(d-1)].transpose(-1, j+2)
                    else:
                        weight = torch.kron(weight.transpose(-1, j+2).contiguous(),p)[...,:-(d-1),:-(d-1)].transpose(-1, j+2)
                break

            out = F.pad(weight, sum((((s-d*(k-1)-1)//2, (s-d*(k-1))//2) for k, d, s in zip(ks, ds, self.shape[2:])), ()))
            if self.stream:
                if i == 0:
                    w = coef[i] * out
                else:
                    w += coef[i] * out
            else:
                cat[...,i] = out

        if not self.stream:
            w = F.linear(cat, coef)

        if self.dims == 1:
            return F.conv1d(x, 
                        w,
                        padding=[(k-1)//2 for k in self.shape[2:]],
                        **self.kwargs)

        return F.conv2d(x, 
                        w,
                        padding=[(k-1)//2 for k in self.shape[2:]],
                        **self.kwargs)


class BaselineSupernet(Supernet):

    def conv2baseline(self, named_modules=None, kernel_sizes=None, dilations=None, relaxation=MixedResults, stream=False, dims=1, separable=True):

        named_modules = self.named_modules() if named_modules is None else named_modules
        named_modules = [(n, m) for n, m in named_modules if hasattr(m, 'kernel_size') and type(m.kernel_size) == tuple and type(m) == Conv(len(m.kernel_size))]

        kernel_sizes = None if kernel_sizes is None else [int2tuple(k, length=dims, allow_other=True) for k in kernel_sizes]
        dilations = None if dilations is None else [int2tuple(d, length=dims, allow_other=True) for d in dilations]

        for name, module in named_modules:
            if 'downsample' in name or 'pool' in name or 'shortcut' in name or 'channel_match' in name:
                continue
            ks = [module.kernel_size] if kernel_sizes is None else kernel_sizes
            ds = [module.dilation] if dilations is None else dilations
            separable = module.out_channels == module.in_channels and separable
            new = relaxation(module.in_channels, 
                             module.out_channels, 
                             ks, 
                             dilations=ds, 
                             stride=module.stride, 
                             groups=module.out_channels if separable else module.groups,
                             stream=stream)
            while True:
                split = name.split('.')
                parent = get_module(self, '.'.join(split[:-1]))
                name = split[-1]
                child = getattr(parent, name)
                setattr(parent, name, new)
                for name, m in self.named_modules():
                    if m == child:
                        break
                else:
                    break

    def remove_module(self, name, named_modules=None):

        named_modules = self.named_modules() if named_modules is None else named_modules
        mods = [(n, m) for n, m in named_modules if name in n]

        for name, module in mods:
            module.remove = True


"""Helper Funcs"""

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from functools import lru_cache

def load_mnist(batch_size, permute=False, seed=1111, input_size=784):
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    flatten = transforms.Lambda(lambda x: x.view(-1, 784))
    if input_size <= 784:
        t = transforms.Lambda(lambda x: x[..., :input_size])
    else:
        t = transforms.Lambda(lambda x: F.pad(x, (0, input_size-784)))

    if permute:
        np.random.seed(seed)
        torch.manual_seed(seed)
        permute = Permute1D(input_size)
        train_transforms = [transforms.ToTensor(), normalize, flatten, permute]
        val_transforms = [transforms.ToTensor(), normalize, flatten, permute]
    else:
        permute = None
        train_transforms = [transforms.ToTensor(), normalize, flatten, t]
        val_transforms = [transforms.ToTensor(), normalize, flatten, t]

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose(train_transforms))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose(val_transforms))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_dataset.__len__(), shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


@lru_cache(maxsize=None)
def atrous_permutation(k, d):

    perm = torch.arange(d*(k-1)+1)
    for i in range(k-1, 0, -1):
        perm[i], perm[d*i] = perm[d*i].item(), perm[i].item()
    return perm


if __name__ == '__main__':
    main()

