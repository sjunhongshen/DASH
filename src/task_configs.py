import time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from itertools import product
from functools import reduce, partial

# import backbone architectures
import networks.resnet as resnet
from networks.wide_resnet import Wide_ResNet, Wide_ResNet_Sep
from networks.tcn import TCN
from networks.deepsea import DeepSEA
from networks.fno import Net2d
from networks.deepcon import DeepCon
from networks.wrn1d import ResNet1D

# import data loaders, task-specific losses and metrics
from data_loaders import load_cifar, load_mnist, load_deepsea, load_darcy_flow, load_psicov, load_music, load_ecg, load_satellite, load_ninapro, load_cosmic, load_spherical, load_fsd
from task_utils import FocalLoss, LpLoss
from task_utils import mask, accuracy, accuracy_onehot, auroc, psicov_mae, ecg_f1, fnr, map_value

# import customized optimizers
# from optimizers import ExpGrad

def get_data(dataset, batch_size, arch, valid_split):
    data_kwargs = None

    if dataset == "your_new_task": # modify this to experiment with a new task
        train_loader, val_loader, test_loader = None, None, None
    elif dataset == "CIFAR10":
        train_loader, val_loader, test_loader = load_cifar(10, batch_size, valid_split=valid_split)
    elif dataset == "CIFAR10-PERM":
        train_loader, val_loader, test_loader = load_cifar(10, batch_size, permute=True, valid_split=valid_split)
    elif dataset == "CIFAR100":
        train_loader, val_loader, test_loader = load_cifar(100, batch_size, valid_split=valid_split)
    elif dataset == "CIFAR100-PERM":
        train_loader, val_loader, test_loader = load_cifar(100, batch_size, permute=True, valid_split=valid_split)
    elif dataset == "MNIST":
        train_loader, val_loader, test_loader = load_mnist(batch_size, valid_split=valid_split)
    elif dataset == "MNIST-PERM":
        train_loader, val_loader, test_loader = load_mnist(batch_size, permute=True, valid_split=valid_split)
    elif dataset == "SPHERICAL":
        train_loader, val_loader, test_loader = load_spherical(batch_size, valid_split=valid_split)
    elif dataset == "DEEPSEA":
        train_loader, val_loader, test_loader = load_deepsea(batch_size, valid_split=valid_split)
    elif dataset == "DARCY-FLOW-5":
        train_loader, val_loader, test_loader, y_normalizer = load_darcy_flow(batch_size, sub = 5, arch = arch, valid_split=valid_split)
        data_kwargs = {"decoder": y_normalizer}
    elif dataset == 'PSICOV':
        train_loader, val_loader, test_loader, _, _ = load_psicov(batch_size, valid_split=valid_split)
    elif dataset[:5] == 'MUSIC':
        if dataset[6] == 'J': length = 255
        elif dataset[6] == 'N': length = 513
        train_loader, val_loader, test_loader = load_music(batch_size, dataset[6:], length=length, valid_split=valid_split)
    elif dataset == "ECG":
        train_loader, val_loader, test_loader = load_ecg(batch_size, valid_split=valid_split)
    elif dataset == "SATELLITE":
        train_loader, val_loader, test_loader = load_satellite(batch_size, valid_split=valid_split)
    elif dataset == "NINAPRO":
        train_loader, val_loader, test_loader = load_ninapro(batch_size, arch, valid_split=valid_split)
    elif dataset == "COSMIC":
        valid_split = True
        train_loader, val_loader, test_loader = load_cosmic(batch_size, valid_split=valid_split)
        data_kwargs = {'transform': mask}
    elif dataset == "FSD":
        train_loader, val_loader, test_loader = load_fsd(batch_size, valid_split=valid_split)

    n_train, n_val, n_test = len(train_loader), len(val_loader) if val_loader is not None else 0, len(test_loader)

    if not valid_split:
        val_loader = test_loader
        n_val = n_test

    return train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs


def get_model(arch, sample_shape, num_classes, config_kwargs, ks = None, ds = None, dropout = None):
    in_channel = sample_shape[1]
    activation = config_kwargs['activation']
    remain_shape = config_kwargs['remain_shape']
    if dropout is None:
        dropout = config_kwargs['dropout']
    pool_k = config_kwargs['pool_k']
    squeeze = config_kwargs['squeeze']

    if len(sample_shape) == 4:

        if arch == 'your_new_arch': # modify this to experiment with a new architecture
            model = None
        elif 'wrn' in arch:
            if 'sep' in arch:
                wrn = Wide_ResNet_Sep
            else:
                wrn = Wide_ResNet
            try:
                splits = arch.split('-')
                model = wrn(int(splits[1]), int(splits[2]), dropout, in_channel=in_channel, num_classes=num_classes, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape, pool_k=pool_k, squeeze=squeeze)
            except IndexError:
                model = wrn(28, 10, 0.3, in_channel=in_channel, num_classes=num_classes, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape, pool_k=pool_k, squeeze=squeeze)
        elif 'convnext' in arch:
            from networks.convnext import convnext_xtiny, convnext_tiny
            model = convnext_xtiny(in_chans=in_channel, num_classes=num_classes, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape)
        elif 'resnet' in arch:
            model = resnet.__dict__[arch](in_channel=in_channel, num_classes=num_classes, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape, pool_k=pool_k, squeeze=squeeze)
        elif 'fno' in arch:
            model = Net2d(12, 32, op=arch[4:], einsum=True, ks = ks, ds = ds)
        elif arch == 'deepcon':
            model = DeepCon(L=128, num_blocks=8, width=16, expected_n_channels=57, no_dilation=False, ks = ks, ds = ds)
    
    else:
        mid_channels = min(4 ** (num_classes // 10 + 1), 64)

        if arch == 'your_new_arch': # modify this to experiment with a new architecture
            model = None
        elif arch == 'TCN':
            model = TCN(in_channel, num_classes, [100] * 8, kernel_size=7, dropout=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape)
        elif arch == 'wrn':
            model = ResNet1D(in_channels = in_channel, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape)
        elif arch == 'deepsea':
            model = DeepSEA(ks = ks, ds = ds)
   
    return model


def get_config(dataset):
    einsum = True
    base, accum = 0.2, 1
    validation_freq = 1
    clip, retrain_clip = 1, -1
    quick_search, quick_retrain = 0.2, 1
    config_kwargs = {'temp': 1, 'arch_retrain_default': None, 'grad_scale': 100, 'activation': None, 'remain_shape': False, 'pool_k': 8, 'squeeze': False, 'dropout': 0}
    
    if dataset == "your_new_task": # modify this to experiment with a new task
        dims, sample_shape, num_classes = None, None, None
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]
        loss = None

        batch_size = 64
        arch_default = 'wrn'


    elif dataset[:5] == "CIFAR":
        dims, sample_shape, num_classes = 2, (1, 3, 32, 32), 10 if dataset in ['CIFAR10', 'CIFAR10-PERM'] else 100
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]
        loss = nn.CrossEntropyLoss()

        batch_size = 64
        arch_default = 'wrn-16-1' 
        config_kwargs['arch_retrain_default'] = 'wrn-16-4' 
        config_kwargs['grad_scale'] = 5000


    elif dataset == 'SPHERICAL':
        dims, sample_shape, num_classes = 2, (1, 3, 60, 60), 100
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15] 
        loss = nn.CrossEntropyLoss() 

        batch_size = 64
        arch_default = 'wrn-16-1'
        config_kwargs['arch_retrain_default'] = 'wrn-16-4' 
        config_kwargs['grad_scale'] = 500


    elif dataset == "DARCY-FLOW-5":
        dims, sample_shape, num_classes = 2, (1, 3, 85, 85), 1
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]  
        loss = LpLoss(size_average=False)
        
        batch_size = 10
        arch_default = 'wrn-16-1'
        config_kwargs['arch_retrain_default'] = 'wrn-16-4'
        config_kwargs['remain_shape'] = config_kwargs['squeeze'] = True 
        

    elif dataset == "PSICOV":
        dims, sample_shape, num_classes = 2, (1, 57, 128, 128), 1
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15]  
        loss = nn.MSELoss(reduction='mean')

        batch_size = 8
        arch_default = 'wrn-16-1'
        config_kwargs['arch_retrain_default'] = 'wrn-16-4'
        config_kwargs['remain_shape']  = True


    elif dataset == "NINAPRO": 
        dims, sample_shape, num_classes = 2, (1, 1, 16, 52), 18
        kernel_choices_default, dilation_choices_default = [3, 5, 7], [1, 3, 7]
        loss = FocalLoss(alpha=1)

        batch_size = 128
        arch_default = 'wrn-16-1'
        config_kwargs['arch_retrain_default'] = 'wrn-16-4'
        config_kwargs['pool_k'] = 0


    elif dataset == "COSMIC":
        dims, sample_shape, num_classes = 2, (1, 1, 128, 128), 1
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15] 
        loss = nn.BCELoss()

        batch_size = 4
        arch_default = 'wrn-16-1'
        config_kwargs['arch_retrain_default'] = 'wrn-16-4'
        config_kwargs['activation'] = 'sigmoid'
        config_kwargs['remain_shape'] = True
        config_kwargs['grad_scale'] = 1000


    elif dataset == 'FSD':
        dims, sample_shape, num_classes = 2, (1, 1, 96, 102), 200
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15] 
        loss = nn.BCEWithLogitsLoss(pos_weight=10 * torch.ones((200, )))
        
        batch_size = 128
        arch_default = 'wrn-16-1'
        config_kwargs['arch_retrain_default'] = 'wrn-16-4'
        config_kwargs['pool_k'] = 0


    elif dataset[:5] == "MNIST":
        dims, sample_shape, num_classes = 1, (1, 1, 784), 10
        kernel_choices_default, dilation_choices_default = [3, 5, 7, 9], [1, 3, 7, 15, 31, 63, 127] 
        loss = F.nll_loss

        batch_size = 256      
        arch_default = 'wrn'
        config_kwargs['activation'] = 'softmax'


    elif dataset[:5] == "MUSIC":
        if dataset[6] == 'J': length = 255 
        elif dataset[6] == 'N': length = 513
        dims, sample_shape, num_classes = 1, (1, 88, length - 1), 88
        kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
        loss = nn.BCELoss()

        batch_size = 4
        arch_default = 'wrn'
        config_kwargs['activation'] = 'sigmoid'
        config_kwargs['remain_shape'] = True

    
    elif dataset == "ECG": 
        dims, sample_shape, num_classes = 1, (1, 1, 1000), 4
        kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
        loss = nn.CrossEntropyLoss()

        batch_size = 1024
        arch_default = 'wrn'
        config_kwargs['activation'] = 'softmax'
        

    elif dataset == "SATELLITE":
        dims, sample_shape, num_classes = 1, (1, 1, 46), 24
        kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
        loss = nn.CrossEntropyLoss()

        batch_size = 256
        arch_default = 'wrn'


    elif dataset == "DEEPSEA":
        dims, sample_shape, num_classes = 1, (1, 4, 1000), 36
        kernel_choices_default, dilation_choices_default = [3, 7, 11, 15, 19], [1, 3, 7, 15]
        loss = nn.BCEWithLogitsLoss(pos_weight=4 * torch.ones((36, )))

        batch_size = 256
        arch_default = 'wrn'  
        config_kwargs['grad_scale'] = 10 


    lr, arch_lr = (1e-2, 5e-3) if config_kwargs['remain_shape'] else (0.1, 0.05) 

    if arch_default[:3] == 'wrn':
        epochs_default, retrain_epochs = 100, 200
        retrain_freq = epochs_default
        opt, arch_opt = partial(torch.optim.SGD, momentum=0.9, nesterov=True), partial(torch.optim.SGD, momentum=0.9, nesterov=True)
        weight_decay = 5e-4 
        
        sched = [60, 120, 160]
        def weight_sched_search(epoch):
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
                    break

            return math.pow(base, optim_factor)

        if dims == 1:
            sched = [30, 60, 90, 120, 160]
        else:
            sched = [60, 120, 160]
        
        def weight_sched_train(epoch):    
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
                    break
                    
            return math.pow(base, optim_factor)

    elif arch_default == 'convnext':
        epochs_default, retrain_epochs, retrain_freq = 100, 300, 100
        opt, arch_opt = torch.optim.AdamW, torch.optim.AdamW
        lr, arch_lr = 4e-3, 1e-2
        weight_decay = 0.05
            
        base_value = lr
        final_value = 1e-6
        niter_per_ep = 392 
        warmup_iters = 0
        epochs = retrain_epochs
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = np.array([final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters]) / base_value

        def weight_sched_search(iter):
            return schedule[iter]
        
        def weight_sched_train(iter):
            return schedule[iter]

    elif arch_default == 'TCN':
        epochs_default, retrain_epochs, retrain_freq = 20, 40, 20
        opt, arch_opt = torch.optim.Adam, torch.optim.Adam
        weight_decay = 1e-4
        
        def weight_sched_search(epoch):
            return base ** (epoch // 10)
        
        def weight_sched_train(epoch):
            return base ** (epoch // 20)

    # arch_opt = ExpGrad

    return dims, sample_shape, num_classes, batch_size, epochs_default, loss, lr, arch_lr, weight_decay, opt, arch_opt, weight_sched_search, weight_sched_train, accum, clip, retrain_clip, validation_freq, retrain_freq,\
    einsum, retrain_epochs, arch_default, kernel_choices_default, dilation_choices_default, quick_search, quick_retrain, config_kwargs


def get_metric(dataset):
    if dataset == "your_new_task": # modify this to experiment with a new task
        return accuracy, np.max
    if dataset[:5] == "CIFAR" or dataset[:5] == "MNIST" or dataset == "SATELLITE" or dataset == "SPHERICAL":
        return accuracy, np.max
    if dataset == "DEEPSEA":
        return auroc, np.max
    if dataset == "DARCY-FLOW-5":
        return LpLoss(size_average=True), np.min
    if dataset == 'PSICOV':
        return psicov_mae(), np.min
    if dataset[:5] == 'MUSIC':
        return nn.BCELoss(), np.min
    if dataset == 'ECG':
        return ecg_f1, np.max
    if dataset == 'NINAPRO':
        return accuracy_onehot, np.max
    if dataset == 'COSMIC':
        return fnr, np.min
    if dataset == 'FSD':
        return map_value, np.max


def get_hp_configs(dataset, n_train):

    epochs = 80
    if n_train < 50:
        subsamping_ratio = 0.2
    elif n_train < 100:
        subsamping_ratio = 0.1
    elif n_train < 500:
        subsamping_ratio = 0.05
    else:
        subsamping_ratio = 0.01

    lrs = 0.1 ** np.arange(1, 4)

    dropout_rates = [0, 0.05]
    wd = [5e-4, 5e-6]
    momentum = [0.9, 0.99]

    if dataset in ['PSICOV', 'COSMIC', 'FSD']: # 2D dense
        lrs = 0.1 ** np.arange(2, 5)
        wd = [5e-5]
        dropout_rates = [0]
        momentum = [0.99]
    configs = list(product(lrs, dropout_rates, wd, momentum))

    return configs, epochs, subsamping_ratio


def get_optimizer(type='SGD', momentum=0.9, weight_decay=5e-4):
    
    return partial(torch.optim.SGD, momentum=momentum, weight_decay=weight_decay, nesterov=(momentum!=0))

