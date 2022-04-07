import numpy as np
import os
import pickle
import gzip
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Customized Data Loaders"""

def load_cifar(num_classes, batch_size, permute=False, seed=1111, valid_split=-1):
    if num_classes == 10:
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    else:
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    if permute:
        np.random.seed(seed)
        torch.manual_seed(seed)
        permute = Permute2D(32, 32)
        train_transforms = [transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), permute, normalize]
        val_transforms = [transforms.ToTensor(), permute, normalize]
    else:
        permute = None
        train_transforms = [transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
        val_transforms = [transforms.ToTensor(), normalize]

    cifar = datasets.CIFAR100 if num_classes == 100 else datasets.CIFAR10

    train_dataset = cifar(root='./data', train=True, transform=transforms.Compose(train_transforms), download=True)
    test_dataset = cifar(root='./data', train=False, transform=transforms.Compose(val_transforms))

    if valid_split > 0:
        valid_dataset = cifar(root='./data', train=True, transform=transforms.Compose(val_transforms))
        train_sampler, valid_sampler = split_dataset(train_dataset, len(test_dataset) / len(train_dataset))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0, pin_memory=True)

    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    if valid_split > 0:
        return train_loader, val_loader, test_loader

    return train_loader, None, test_loader


def load_mnist(batch_size, permute=False, seed=1111, valid_split=-1):
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    flatten = transforms.Lambda(lambda x: x.view(-1, 784))
    if permute:
        np.random.seed(seed)
        torch.manual_seed(seed)
        permute = Permute1D(784)
        train_transforms = [transforms.ToTensor(), normalize, flatten, permute]
        val_transforms = [transforms.ToTensor(), normalize, flatten, permute]
    else:
        permute = None
        train_transforms = [transforms.ToTensor(), normalize, flatten]
        val_transforms = [transforms.ToTensor(), normalize, flatten]

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose(train_transforms))
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose(val_transforms))

    if valid_split > 0:
        valid_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.Compose(val_transforms))
        train_sampler, valid_sampler = split_dataset(train_dataset, len(test_dataset) / len(train_dataset))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=4, pin_memory=True)

    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)     
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    if valid_split > 0:
        return train_loader, val_loader, test_loader

    return train_loader, None, test_loader


def load_spherical(batch_size, valid_split=-1):

    with gzip.open('./data/s2_cifar100.gz', 'rb') as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["train"]["images"][:, None, :, :].astype(np.float32)).squeeze()
    train_labels = torch.from_numpy(
        dataset["train"]["labels"].astype(np.int64))

    test_data = torch.from_numpy(
        dataset["test"]["images"][:, None, :, :].astype(np.float32)).squeeze()
    test_labels = torch.from_numpy(
        dataset["test"]["labels"].astype(np.int64))

    if valid_split > 0:
        test_size = len(test_labels)
        shuffle_pid = np.random.permutation(len(train_labels))
        train_data = train_data[shuffle_pid]
        train_labels = train_labels[shuffle_pid]
        train_data, val_data = train_data[:-test_size], train_data[-test_size:]
        train_labels, val_labels = train_labels[:-test_size], train_labels[-test_size:]

        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if valid_split > 0:
        return train_loader, val_loader, test_loader

    return train_loader, None, test_loader


def load_deepsea(batch_size, one_hot = True, valid_split=-1):
    filename = './data/deepsea_filtered.npz'
    data = np.load(filename)

    if valid_split > 0:
        if one_hot:
            x_train = torch.from_numpy(data['x_train']).transpose(-1, -2).float()  
        else:
            x_train = torch.from_numpy(np.argmax(data['x_train'], axis=2)).unsqueeze(-2).float()
        y_train = torch.from_numpy(data['y_train']).float() 
        if one_hot:
            x_val = torch.from_numpy(data['x_val']).transpose(-1, -2).float()   # shape = (2490, 1000, 4)
        else:
            x_val = torch.from_numpy(np.argmax(data['x_val'], axis=2)).unsqueeze(-2).float() 
        y_val = torch.from_numpy(data['y_val']).float()  # shape = (2490, 36)

    else:
        if one_hot:
            x_train = torch.from_numpy(np.concatenate((data['x_train'], data['x_val']), axis=0)).transpose(-1, -2).float()  
        else:
            x_train = torch.from_numpy(np.argmax(np.concatenate((data['x_train'], data['x_val']), axis=0), axis=2)).unsqueeze(-2).float()
        y_train = torch.from_numpy(np.concatenate((data['y_train'], data['y_val']), axis=0)).float() 

    if one_hot:
        x_test = torch.from_numpy(data['x_test']).transpose(-1, -2).float()  # shape = (149400, 1000, 4)
    else:
        x_test = torch.from_numpy(np.argmax(data['x_test'], axis=2)).unsqueeze(-2).float()
    y_test = torch.from_numpy(data['y_test']).float()   # shape = (149400, 36)
    
    if valid_split > 0:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size = 512, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size = 512, shuffle=True, num_workers=4, pin_memory=True)
        
        return train_loader, val_loader, test_loader
    else:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size = 512, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, None, test_loader


def load_darcy_flow(batch_size, sub = 5, arch = None, valid_split=-1):
    ntrain = 1000
    ntest = 100
    r = sub # 5, 3, 2, 1
    h = int(((421 - 1)/r) + 1)
    s = h
    TRAIN_PATH = os.path.join("./data", 'piececonst_r421_N1024_smooth1.mat')
    TEST_PATH = os.path.join("./data", 'piececonst_r421_N1024_smooth2.mat')

    if not os.path.isfile(TRAIN_PATH):
        # print("downloading training data")
        with open(TRAIN_PATH, 'wb') as f:
            f.write(requests.get("https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth1.mat").content)
    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
    y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

    if valid_split > 0:
        x_train = reader.read_field('coeff')[:900,::r,::r][:,:s,:s]
        y_train = reader.read_field('sol')[:900,::r,::r][:,:s,:s]
        x_val = reader.read_field('coeff')[900:1000,::r,::r][:,:s,:s]
        y_val = reader.read_field('sol')[900:1000,::r,::r][:,:s,:s]
        ntrain = 900
        nval = len(y_val)

    if not os.path.isfile(TEST_PATH):
        # print("downloading test data")
        with open(TEST_PATH, 'wb') as f:
            f.write(requests.get("https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth2.mat").content)
    reader.load_file(TEST_PATH)
    x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
    y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    y_test = y_normalizer.encode(y_test)

    grids = []
    grids.append(np.linspace(0, 1, s))
    grids.append(np.linspace(0, 1, s))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1,s,s,2)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain,s,s,1), grid.repeat(ntrain,1,1,1)], dim=3)
    x_test = torch.cat([x_test.reshape(ntest,s,s,1), grid.repeat(ntest,1,1,1)], dim=3)

    if valid_split > 0:
        x_val = x_normalizer.encode(x_val)
        y_val = y_normalizer.encode(y_val)
        x_val = torch.cat([x_val.reshape(nval,s,s,1), grid.repeat(nval,1,1,1)], dim=3)

    if arch[:3] != 'fno':
        x_train = x_train.permute(0, 3, 1, 2)
        x_test = x_test.permute(0, 3, 1, 2)
        if valid_split > 0:
            x_val = x_val.permute(0, 3, 1, 2)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if valid_split > 0:
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader, test_loader, y_normalizer

    return train_loader, None, test_loader, y_normalizer


def load_psicov(batch_size, valid_split=-1):
    training_window = 128
    expected_n_channels = 57
    pad_size = 10

    all_feat_paths = ['./data/protein/deepcov/features/', './data/protein/psicov/features/', './data/protein/cameo/features/']
    all_dist_paths = ['./data/protein/deepcov/distance/', './data/protein/psicov/distance/', './data/protein/cameo/distance/']

    deepcov_list = load_list('./data/protein/deepcov.lst', -1)

    length_dict = {}
    for pdb in deepcov_list:
        (ly, seqy, cb_map) = np.load('./data/protein/deepcov/distance/' + pdb + '-cb.npy', allow_pickle = True)
        length_dict[pdb] = ly

    psicov_list = load_list('./data/protein/psicov.lst')
    psicov_length_dict = {}
    for pdb in psicov_list:
        (ly, seqy, cb_map) = np.load('./data/protein/psicov/distance/' + pdb + '-cb.npy', allow_pickle = True)
        psicov_length_dict[pdb] = ly

    # print('')
    # print('Split into training and validation set..')
    train_pdbs = deepcov_list

    if valid_split > 0:
        test_pdbs = psicov_list[:int(0.5 * len(psicov_list))]
        valid_pdbs = psicov_list[int(0.5 * len(psicov_list)):]
        valid_dataset = PDNetDataset(valid_pdbs, all_feat_paths, all_dist_paths, 512, pad_size, 1, expected_n_channels, label_engineering = None)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    else:
        test_pdbs = psicov_list

    # print('Total test proteins : ', len(test_pdbs))
    # print('Total training proteins   : ', len(train_pdbs))

    train_dataset = PDNetDataset(train_pdbs, all_feat_paths, all_dist_paths, training_window, pad_size, batch_size, expected_n_channels, label_engineering = '16.0')
    test_dataset = PDNetDataset(test_pdbs, all_feat_paths, all_dist_paths, 512, pad_size, 1, expected_n_channels, label_engineering = None)
            
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    if valid_split > 0:
        return train_loader, val_loader, test_loader, psicov_list, psicov_length_dict

    return train_loader, None, test_loader, psicov_list, psicov_length_dict

import scipy.io

def load_music(batch_size, dataset, length=65, valid_split=-1):
    if dataset == "JSB":
        # print('loading JSB data...')
        data = scipy.io.loadmat('./data/music/JSB_Chorales.mat')
    elif dataset == "Muse":
        # print('loading Muse data...')
        data = scipy.io.loadmat('./data/music/MuseData.mat')
    elif dataset == "Nott":
        # print('loading Nott data...')
        data = scipy.io.loadmat('./data/music/Nottingham.mat')
    elif dataset == "Piano":
        # print('loading Piano data...')
        data = scipy.io.loadmat('./data/music/Piano_midi.mat')

    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]

    x_train, y_train = [], []

    if valid_split > 0:
        for data in X_train:
            for data_i in data:
                if data_i.shape[0] < length:
                    data_i = np.pad(data_i, ((0, length - data_i.shape[0]), (0, 0)))
                else:
                    data_i = data_i[:length, :]

                x_train.append(data_i[:-1].T)
                y_train.append(data_i[1:].T)

        x_val, y_val = [], []
        for data in X_valid:
            for data_i in data:
                if data_i.shape[0] < length:
                    data_i = np.pad(data_i, ((0, length - data_i.shape[0]), (0, 0)))
                else:
                    data_i = data_i[:length, :]

                x_val.append(data_i[:-1].T)
                y_val.append(data_i[1:].T)

        x_val = torch.from_numpy(np.stack(x_val, axis=0)).float()
        y_val = torch.from_numpy(np.stack(y_val, axis=0)).float()
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    else:
        for data in [X_train, X_valid]:
            for data_i in data:
                if data_i.shape[0] < length:
                    data_i = np.pad(data_i, ((0, length - data_i.shape[0]), (0, 0)))
                else:
                    data_i = data_i[:length, :]

                x_train.append(data_i[:-1].T)
                y_train.append(data_i[1:].T)

    x_test, y_test = [], []
    for data_i in X_test:
        if data_i.shape[0] < length:
            data_i = np.pad(data_i, ((0, length - data_i.shape[0]), (0, 0)))
        else:
            data_i = data_i[:length, :]
        x_test.append(data_i[:-1].T)
        y_test.append(data_i[1:].T)

    x_train = torch.from_numpy(np.stack(x_train, axis=0)).float()
    y_train = torch.from_numpy(np.stack(y_train, axis=0)).float()
    x_test = torch.from_numpy(np.stack(x_test, axis=0)).float()
    y_test = torch.from_numpy(np.stack(y_test, axis=0)).float()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if valid_split > 0:
        return train_loader, val_loader, test_loader

    return train_loader, None, test_loader


from sklearn.model_selection import train_test_split

def load_ecg(batch_size, valid_split=-1):
    window_size = 1000
    stride = 500

    # read pkl
    with open('./data/challenge2017.pkl', 'rb') as fin:
        res = pickle.load(fin)
    ## scale data
    all_data = res['data']
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std
    ## encode label
    all_label = []
    for i in res['label']:
        if i == 'N':
            all_label.append(0)
        elif i == 'A':
            all_label.append(1)
        elif i == 'O':
            all_label.append(2)
        elif i == '~':
            all_label.append(3)
    all_label = np.array(all_label)

    # split train test
    if valid_split > 0:
        X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_label, test_size=0.2, random_state=0)
        X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)
        X_val, Y_val = slide_and_cut(X_val, Y_val, window_size=window_size, stride=stride)

        shuffle_pid = np.random.permutation(Y_val.shape[0])
        X_val = X_val[shuffle_pid]
        Y_val = Y_val[shuffle_pid]
        X_val = torch.from_numpy(np.expand_dims(X_val, 1)).float()
        Y_val = torch.from_numpy(Y_val).long()

        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    else:    
        X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_label, test_size=0.1, random_state=0)

    X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=window_size, stride=stride)
    X_test, Y_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride)

    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    shuffle_pid = np.random.permutation(Y_test.shape[0])
    X_test = X_test[shuffle_pid]
    Y_test = Y_test[shuffle_pid]

    X_train = torch.from_numpy(np.expand_dims(X_train, 1)).float()
    X_test = torch.from_numpy(np.expand_dims(X_test, 1)).float()
    Y_train = torch.from_numpy(Y_train).long()
    Y_test = torch.from_numpy(Y_test).long()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if valid_split > 0:
        return train_loader, val_loader, test_loader

    return train_loader, None, test_loader


def load_satellite(batch_size, valid_split=-1):
    path = './data'
    train_file = os.path.join(path, 'satellite_train.npy')
    test_file = os.path.join(path, 'satellite_test.npy')

    X_training, Y_training = np.load(train_file, allow_pickle=True)[()]['data'], np.load(train_file,allow_pickle=True)[()]['label']
    X_test, Y_test = np.load(test_file, allow_pickle=True)[()]['data'], np.load(test_file, allow_pickle=True)[()]['label']
    Y_training = Y_training - 1
    Y_test = Y_test - 1

    if valid_split > 0:
        shuffle_pid = np.random.permutation(Y_training.shape[0])
        X_training = X_training[shuffle_pid]
        Y_training = Y_training[shuffle_pid]

        len_val = len(Y_test)
        X_training, Y_training = X_training[:-len_val], Y_training[:-len_val]
        X_validation, Y_validation = X_training[-len_val:], Y_training[-len_val:]

        X_validation = torch.from_numpy(np.expand_dims(X_validation, 1)).float()
        Y_validation = torch.from_numpy(Y_validation).long()

        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_validation, Y_validation), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    X_training = torch.from_numpy(np.expand_dims(X_training, 1)).float()
    Y_training = torch.from_numpy(Y_training).long()
    X_test = torch.from_numpy(np.expand_dims(X_test, 1)).float()
    Y_test = torch.from_numpy(Y_test).long()

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_training, Y_training), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if valid_split > 0:
        return train_loader, val_loader, test_loader

    return train_loader, None, test_loader


def load_ninapro(batch_size, arch, valid_split=-1):

    path = './data/ninaPro'
    train_data = np.load(os.path.join(path, 'ninapro_train.npy'),
                                 encoding="bytes", allow_pickle=True)
    train_labels = np.load(os.path.join(path, 'label_train.npy'), encoding="bytes", allow_pickle=True).astype(np.int8)

    valid_data = np.load(os.path.join(path, 'ninapro_val.npy'),
                                 encoding="bytes", allow_pickle=True)
    valid_labels = np.load(os.path.join(path, 'label_val.npy'), encoding="bytes", allow_pickle=True).astype(np.int8)

    test_data = np.load(os.path.join(path, 'ninapro_test.npy'),
                                 encoding="bytes", allow_pickle=True)
    test_labels = np.load(os.path.join(path, 'label_test.npy'), encoding="bytes", allow_pickle=True).astype(np.int8)
    
    if valid_split > 0:
        valid_labels = np.eye(18)[valid_labels]
        valid_data = torch.from_numpy(valid_data).float().permute(0, 2, 1)
        valid_labels = torch.from_numpy(valid_labels).float()
    else:
        train_data = np.concatenate((train_data, valid_data), axis=0)
        train_labels = np.concatenate((train_labels, valid_labels), axis=0).astype(np.int8)
    
    train_labels = np.eye(18)[train_labels]
    test_labels = np.eye(18)[test_labels]

    train_data = torch.from_numpy(train_data).float().permute(0, 2, 1)
    train_labels = torch.from_numpy(train_labels).float()#.unsqueeze(-1)
    test_data = torch.from_numpy(test_data).float().permute(0, 2, 1)
    test_labels = torch.from_numpy(test_labels).float()#.unsqueeze(-1)

    if 'wrn' in arch or 'resnet' in arch:
        train_data = train_data.unsqueeze(1)
        test_data = test_data.unsqueeze(1)
        if valid_split > 0:
            valid_data = valid_data.unsqueeze(1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_labels), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if valid_split > 0:
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_data, valid_labels), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader, test_loader

    return train_loader, None, test_loader


def load_cosmic(batch_size, valid_split=-1):
    aug_sky=[-0.9, 3]
    path = './data/cosmic'
    train_dirs = np.load(os.path.join(path, 'train_dirs.npy'),allow_pickle =True)
    test_dirs = np.load(os.path.join(path, 'test_dirs.npy'),allow_pickle =True)

    if valid_split > 0:
        test_size = len(test_dirs)
        shuffle_pid = np.random.permutation(train_dirs.shape[0])
        train_dirs = train_dirs[shuffle_pid]
        train_dirs, val_dirs = train_dirs[:-test_size], train_dirs[-test_size:]
        data_val = PairedDatasetImagePath(val_dirs[::], aug_sky[0], aug_sky[1], part=None)
        val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    data_train = PairedDatasetImagePath(train_dirs[::], aug_sky[0], aug_sky[1], part=None)
    data_test = PairedDatasetImagePath(test_dirs[::], aug_sky[0], aug_sky[1], part=None)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)       
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if valid_split > 0:
        return train_loader, val_loader, test_loader

    return train_loader, None, test_loader


def load_fsd(batch_size, valid_split=-1):
    from data.audio_dataset import BackgroundAddMixer, UseMixerWithProb, get_transforms_fsd_chunks, SpectrogramDataset, FSD50kEvalDataset

    feature='mel'

    path = './data'
    meta_root = os.path.join(path, "chunks")
    train_manifest = "tr.csv"
    val_manifest = "val.csv"
    label_map = "lbl_map.json"
    test_manifest = 'eval.csv'

    train_manifest = os.path.join(meta_root, train_manifest)
    val_manifest = os.path.join(meta_root, val_manifest)
    test_manifest = os.path.join(meta_root, test_manifest)
    label_map = os.path.join(meta_root, label_map)

    bg_files = os.path.join(path, "noise_22050")

    if feature == 'mel':
        audio_config = {
            'feature': 'melspectrogram',
            'sample_rate': 22050,
            'min_duration': 1,
            'bg_files': bg_files,
        }
    elif feature == 'raw':
        audio_config = {
            'feature': 'spectrogram',
            'n_fft': 441,
            'hop_len': 220,
            'normalize': False,
            'sample_rate': 22050,
            'min_duration': 1,
            'bg_files': bg_files,
        }
    else:
        raise KeyError

    mixer = BackgroundAddMixer()
    train_mixer = UseMixerWithProb(mixer, 0.75)
    train_transforms = get_transforms_fsd_chunks(True, 101)
    val_transforms = get_transforms_fsd_chunks(False, 101)
    #precision = 16

    train_set = SpectrogramDataset(train_manifest,
                                        label_map,
                                        audio_config,
                                        mode="multilabel", augment=True,
                                        mixer=train_mixer,
                                        transform=train_transforms)

    val_set = FSD50kEvalDataset(val_manifest, label_map,
                                     audio_config,
                                     transform=val_transforms
                                     )

    test_set = FSD50kEvalDataset(test_manifest, label_map,
                                 audio_config,
                                 transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = val_set
    test_loader = test_set

    if valid_split > 0:
        return train_loader, val_loader, test_loader

    return train_loader, None, test_loader


"""Hepler Funcs"""

def split_dataset(train_dataset, valid_split):
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_split * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler

def bitreversal_permutation(n):
    """Return the bit reversal permutation used in FFT.
    Parameter:
        n: integer, must be a power of 2.
    Return:
        perm: bit reversal permutation, numpy array of size n
    """
    m = int(math.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    perm = np.arange(n).reshape(n, 1)
    for i in range(m):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    return torch.tensor(perm.squeeze(0))

class Permute2D(nn.Module):

    def __init__(self, row, col):

        super().__init__()
        self.rowperm = torch.LongTensor(bitreversal_permutation(row))
        self.colperm = torch.LongTensor(bitreversal_permutation(col))

    def forward(self, tensor):

        return tensor[:,self.rowperm][:,:,self.colperm]

class Permute1D(nn.Module):

    def __init__(self, length):

        super().__init__()
        self.permute = torch.Tensor(np.random.permutation(length).astype(np.float64)).long()

    def forward(self, tensor):

        return tensor[:,self.permute]


class PairedDatasetImagePath(torch.utils.data.Dataset):
    def __init__(self, paths, skyaug_min=0, skyaug_max=0, part=None, f_val=0.1, seed=1):
        """ custom pytorch dataset class to load deepCR-mask training data
        :param paths: (list) list of file paths to (3, W, H) images: image, cr, ignore.
        :param skyaug_min: [float, float]. If sky is provided, use random sky background in the range
          [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers to allow the trained model to adapt to a
          wider range of sky background or equivalently exposure time. Remedy the fact that exposure time in the
          training set is discrete and limited.
        :param skyaug_min: float. subtract maximum amount of abs(skyaug_min) * sky_level as data augmentation
        :param skyaug_max: float. add maximum amount of skyaug_max * sky_level as data augmentation
        :param part: either 'train' or 'val'.
        :param f_val: percentage of dataset reserved as validation set.
        :param seed: fix numpy random seed to seed, for reproducibility.
        """
        assert 0 < f_val < 1
        np.random.seed(seed)
        n_total = len(paths)
        n_train = int(n_total * (1 - f_val)) #int(len * (1 - f_val)) JK
        f_test = f_val
        n_search = int(n_total * (1 - f_val - f_test))

        if part == 'train':
            s = np.s_[:max(1, n_train)]
        elif part == 'test':
            s = np.s_[min(n_total - 1, n_train):]
        else:
            s = np.s_[0:]

        self.paths = paths[s]
        self.skyaug_min = skyaug_min
        self.skyaug_max = skyaug_max

    def __len__(self):
        return len(self.paths)

    def get_skyaug(self, i):
        """
        Return the amount of background flux to be added to image
        The original sky background should be saved in sky.npy in each sub-directory
        Otherwise always return 0
        :param i: index of file
        :return: amount of flux to add to image
        """
        path = os.path.split(self.paths[i])[0]
        sky_path = os.path.join('./data', path[2:], 'sky.npy') #JK
        if os.path.isfile(sky_path):
            f_img = self.paths[i].split('/')[-1]
            sky_idx = int(f_img.split('_')[0])
            sky = np.load(sky_path)[sky_idx-1]
            return sky * np.random.uniform(self.skyaug_min, self.skyaug_max)
        else:
            return 0

    def __getitem__(self, i):
        data = np.load(os.path.join('./data', self.paths[i][2:]))
        image = data[0]
        mask = data[1]
        if data.shape[0] == 3:
            ignore = data[2]
        else:
            ignore = np.zeros_like(data[0])
        # try:#JK
        skyaug = self.get_skyaug(i)
        #crop to 128*128
        image, mask, ignore = get_fixed_crop([image, mask, ignore], 128, 128)

        return np.expand_dims(image + skyaug, 0).astype(np.float32), np.expand_dims(mask,0).astype(np.float32), np.expand_dims(ignore,0).astype(np.float32)

def get_random_crop(images, crop_height, crop_width):

    max_x = images[0].shape[1] - crop_width
    max_y = images[0].shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crops = []
    for image in images:
        crop = image[y: y + crop_height, x: x + crop_width]
        crops.append(crop)

    return crops

def get_fixed_crop(images, crop_height, crop_width):

    x = 64
    y = 64

    crops = []
    for image in images:
        crop = image[y: y + crop_height, x: x + crop_width]
        crops.append(crop)

    return crops

import h5py

import operator
from functools import reduce
from functools import partial

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class DistGenerator():
    def __init__(self, pdb_id_list, features_path, distmap_path, dim, pad_size, batch_size, expected_n_channels, label_engineering = None):
        self.pdb_id_list = pdb_id_list
        self.features_path = features_path
        self.distmap_path = distmap_path
        self.dim = dim
        self.pad_size = pad_size
        self.batch_size = batch_size
        self.expected_n_channels = expected_n_channels
        self.label_engineering = label_engineering

    def on_epoch_begin(self):
        self.indexes = np.arange(len(self.pdb_id_list))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(len(self.pdb_id_list) / self.batch_size)

    def __getitem__(self, index):
        batch_list = self.pdb_id_list[index * self.batch_size: (index + 1) * self.batch_size]
        X, Y = get_input_output_dist(batch_list, self.features_path, self.distmap_path, self.pad_size, self.dim, self.expected_n_channels)
        if self.label_engineering is None:
            return X, Y
        if self.label_engineering == '100/d':
            return X, 100.0 / Y
        try:
            t = float(self.label_engineering)
            Y[Y > t] = t
        except ValueError:
            print('ERROR!! Unknown label_engineering parameter!!')
            return 
        return X, Y

class PDNetDataset(torch.utils.data.Dataset):
    def __init__(self, pdb_id_list, features_path, distmap_path, dim, 
        pad_size, batch_size, expected_n_channels, label_engineering = None):

        self.distGenerator = DistGenerator(
            pdb_id_list, features_path, distmap_path, dim, pad_size, 
            1, expected_n_channels, label_engineering) # Don't use batch_size

    def __len__(self):
        return self.distGenerator.__len__()

    def __getitem__(self, index):
        X, Y = self.distGenerator.__getitem__(index)
        X = X[0, :, :, :].transpose(2, 0, 1)
        Y = Y[0, :, :, :].transpose(2, 0, 1)
        X = np.nan_to_num(X, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)
        return X, Y

import pickle
import random

def load_list(file_lst, max_items = 1000000):
    if max_items < 0:
        max_items = 1000000
    protein_list = []
    f = open(file_lst, 'r')
    for l in f.readlines():
        protein_list.append(l.strip().split()[0])
    if (max_items < len(protein_list)):
        protein_list = protein_list[:max_items]
    return protein_list

def summarize_channels(x, y):
    # print(' Channel        Avg        Max        Sum')
    for i in range(len(x[0, 0, :])):
        (m, s, a) = (x[:, :, i].flatten().max(), x[:, :, i].flatten().sum(), x[:, :, i].flatten().mean())
        print(' %7s %10.4f %10.4f %10.1f' % (i+1, a, m, s))
    # print("      Ymin = %.2f  Ymean = %.2f  Ymax = %.2f" % (y.min(), y.mean(), y.max()) )

def get_bulk_output_contact_maps(pdb_id_list, all_dist_paths, OUTL):
    YY = np.full((len(pdb_id_list), OUTL, OUTL, 1), 100.0)
    for i, pdb in enumerate(pdb_id_list):
        Y = get_map(pdb, all_dist_paths)
        ly = len(Y[:, 0])
        assert ly <= OUTL
        YY[i, :ly, :ly, 0] = Y
    if np.any(np.isnan(Y)):
        print('')
        print('WARNING:')
        print('Some pdbs in the following list have NaNs in their distances:', pdb_id_list)
        np.seterr(invalid='ignore')
    YY[ YY < 8.0 ] = 1.0
    YY[ YY >= 8.0 ] = 0.0
    return YY.astype(np.float32)

def get_bulk_output_dist_maps(pdb_id_list, all_dist_paths, OUTL):
    YY = np.full((len(pdb_id_list), OUTL, OUTL, 1), np.inf)
    for i, pdb in enumerate(pdb_id_list):
        Y = get_map(pdb, all_dist_paths)
        ly = len(Y[:, 0])
        assert ly <= OUTL
        YY[i, :ly, :ly, 0] = Y
    return YY.astype(np.float32)

def get_input_output_dist(pdb_id_list, all_feat_paths, all_dist_paths, pad_size, OUTL, expected_n_channels):
    XX = np.full((len(pdb_id_list), OUTL, OUTL, expected_n_channels), 0.0)
    YY = np.full((len(pdb_id_list), OUTL, OUTL, 1), 100.0)
    for i, pdb in enumerate(pdb_id_list):
        X = get_feature(pdb, all_feat_paths, expected_n_channels)
        assert len(X[0, 0, :]) == expected_n_channels
        Y0 = get_map(pdb, all_dist_paths, len(X[:, 0, 0]))
        assert len(X[:, 0, 0]) >= len(Y0[:, 0])
        if len(X[:, 0, 0]) != len(Y0[:, 0]):
            print('')
            print('WARNING!! Different len(X) and len(Y) for ', pdb, len(X[:, 0, 0]), len(Y0[:, 0]))
        l = len(X[:, 0, 0])
        Y = np.full((l, l), np.nan)
        Y[:len(Y0[:, 0]), :len(Y0[:, 0])] = Y0
        Xpadded = np.zeros((l + pad_size, l + pad_size, len(X[0, 0, :])), dtype=np.float32)
        Xpadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2), :] = X
        Ypadded = np.full((l + pad_size, l + pad_size), 100.0, dtype=np.float32)
        Ypadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2)] = Y
        l = len(Xpadded[:, 0, 0])
        if l <= OUTL:
            XX[i, :l, :l, :] = Xpadded
            YY[i, :l, :l, 0] = Ypadded
        else:
            rx = random.randint(0, l - OUTL)
            ry = random.randint(0, l - OUTL)
            assert rx + OUTL <= l
            assert ry + OUTL <= l
            XX[i, :, :, :] = Xpadded[rx:rx+OUTL, ry:ry+OUTL, :]
            YY[i, :, :, 0] = Ypadded[rx:rx+OUTL, ry:ry+OUTL]
    return XX.astype(np.float32), YY.astype(np.float32)

def get_input_output_bins(pdb_id_list, all_feat_paths, all_dist_paths, pad_size, OUTL, expected_n_channels, bins):
    XX = np.full((len(pdb_id_list), OUTL, OUTL, expected_n_channels), 0.0)
    YY = np.full((len(pdb_id_list), OUTL, OUTL, len(bins)), 0.0)
    for i, pdb in enumerate(pdb_id_list):
        X = get_feature(pdb, all_feat_paths, expected_n_channels)
        assert len(X[0, 0, :]) == expected_n_channels
        Y0 = dist_map_to_bins(get_map(pdb, all_dist_paths, len(X[:, 0, 0])), bins)
        assert len(X[:, 0, 0]) >= len(Y0[:, 0])
        if len(X[:, 0, 0]) != len(Y0[:, 0]):
            print('')
            print('WARNING!! Different len(X) and len(Y) for ', pdb, len(X[:, 0, 0]), len(Y0[:, 0]))
        l = len(X[:, 0, 0])
        Y = np.full((l, l, len(Y0[0, 0, :])), np.nan)
        Y[:len(Y0[:, 0]), :len(Y0[:, 0]), :] = Y0
        Xpadded = np.zeros((l + pad_size, l + pad_size, len(X[0, 0, :])))
        Xpadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2), :] = X
        Ypadded = np.full((l + pad_size, l + pad_size, len(bins)), 0.0)
        Ypadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2), :] = Y
        l = len(Xpadded[:, 0, 0])
        if l <= OUTL:
            XX[i, :l, :l, :] = Xpadded
            YY[i, :l, :l, :] = Ypadded
        else:
            rx = random.randint(0, l - OUTL)
            ry = random.randint(0, l - OUTL)
            assert rx + OUTL <= l
            assert ry + OUTL <= l
            XX[i, :, :, :] = Xpadded[rx:rx+OUTL, ry:ry+OUTL, :]
            YY[i, :, :, :] = Ypadded[rx:rx+OUTL, ry:ry+OUTL, :]
    return XX.astype(np.float32), YY.astype(np.float32)

def get_sequence(pdb, feature_file):
    features = pickle.load(open(feature_file, 'rb'))
    return features['seq']

def get_feature(pdb, all_feat_paths, expected_n_channels):
    features = None
    for path in all_feat_paths:
        if os.path.exists(path + pdb + '.pkl'):
            features = pickle.load(open(path + pdb + '.pkl', 'rb'))
    if features == None:
        print('Expected feature file for', pdb, 'not found at', all_feat_paths)
        exit(1)
    l = len(features['seq'])
    seq = features['seq']
    # Create X and Y placeholders
    X = np.full((l, l, expected_n_channels), 0.0)
    # Add secondary structure
    ss = features['ss']
    assert ss.shape == (3, l)
    fi = 0
    for j in range(3):
        a = np.repeat(ss[j].reshape(1, l), l, axis = 0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    # Add PSSM
    pssm = features['pssm']
    assert pssm.shape == (l, 22)
    for j in range(22):
        a = np.repeat(pssm[:, j].reshape(1, l), l, axis = 0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    # Add SA
    sa = features['sa']
    assert sa.shape == (l, )
    a = np.repeat(sa.reshape(1, l), l, axis = 0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    # Add entrophy
    entropy = features['entropy']
    assert entropy.shape == (l, )
    a = np.repeat(entropy.reshape(1, l), l, axis = 0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    # Add CCMpred
    ccmpred = features['ccmpred']
    assert ccmpred.shape == ((l, l))
    X[:, :, fi] = ccmpred
    fi += 1
    # Add  FreeContact
    freecon = features['freecon']
    assert freecon.shape == ((l, l))
    X[:, :, fi] = freecon
    fi += 1
    # Add potential
    potential = features['potential']
    assert potential.shape == ((l, l))
    X[:, :, fi] = potential
    fi += 1
    assert fi == expected_n_channels
    assert X.max() < 100.0
    assert X.min() > -100.0
    return X

def get_map(pdb, all_dist_paths, expected_l = -1):
    seqy = None
    mypath = ''
    for path in all_dist_paths:
        if os.path.exists(path + pdb + '-cb.npy'):
            mypath = path + pdb + '-cb.npy'
            (ly, seqy, cb_map) = np.load(path + pdb + '-cb.npy', allow_pickle = True)
    if seqy == None:
        print('Expected distance map file for', pdb, 'not found at', all_dist_paths)
        exit(1)
    if 'cameo' not in mypath and expected_l > 0:
        assert expected_l == ly
        assert cb_map.shape == ((expected_l, expected_l))
    Y = cb_map
    # Only CAMEO dataset has this issue
    if 'cameo' not in mypath:
        assert not np.any(np.isnan(Y))
    if np.any(np.isnan(Y)):
        np.seterr(invalid='ignore')
        print('')
        print('WARNING!! Some values in the pdb structure of', pdb, 'l = ', ly, 'are missing or nan! Indices are: ', np.where(np.isnan(np.diagonal(Y))))
    Y[Y < 1.0] = 1.0
    Y[0, 0] = Y[0, 1]
    Y[ly-1, ly-1] = Y[ly-1, ly-2]
    for q in range(1, ly-1):
        if np.isnan(Y[q, q]):
            continue
        if np.isnan(Y[q, q-1]) and np.isnan(Y[q, q+1]):
            Y[q, q] = 1.0
        elif np.isnan(Y[q, q-1]):
            Y[q, q] = Y[q, q+1]
        elif np.isnan(Y[q, q+1]):
            Y[q, q] = Y[q, q-1]
        else:
            Y[q, q] = (Y[q, q-1] + Y[q, q+1]) / 2.0
    assert np.nanmax(Y) <= 500.0
    assert np.nanmin(Y) >= 1.0
    return Y

def save_dist_rr(pdb, pred_matrix, feature_file, file_rr):
    sequence = get_sequence(pdb, feature_file)
    rr = open(file_rr, 'w')
    rr.write(sequence + "\n")
    P = np.copy(pred_matrix)
    L = len(P[:])
    for j in range(0, L):
        for k in range(j, L):
            P[j, k] = (P[k, j, 0] + P[j, k, 0]) / 2.0
    for j in range(0, L):
        for k in range(j, L):
            if abs(j - k) < 5:
                continue
            rr.write("%i %i %0.3f %.3f 1\n" %(j+1, k+1, P[j][k], P[j][k]) )
    rr.close()
    print('Written RR ' + file_rr + ' !')

def save_contacts_rr(pdb, all_feat_paths, pred_matrix, file_rr):
    for path in all_feat_paths:
        if os.path.exists(path + pdb + '.pkl'):
            features = pickle.load(open(path + pdb + '.pkl', 'rb'))
    if features == None:
        print('Expected feature file for', pdb, 'not found at', all_feat_paths)
        exit(1)
    sequence = features['seq']
    rr = open(file_rr, 'w')
    rr.write(sequence + "\n")
    P = np.copy(pred_matrix)
    L = len(P[:])
    for j in range(0, L):
        for k in range(j, L):
            P[j, k] = (P[k, j, 0] + P[j, k, 0]) / 2.0
    for j in range(0, L):
        for k in range(j, L):
            if abs(j - k) < 5:
                continue
            rr.write("%i %i 0 8 %.5f\n" %(j+1, k+1, (P[j][k])) )
    rr.close()
    print('Written RR ' + file_rr + ' !')

def dist_map_to_bins(Y, bins):
    L = len(Y[:, 0])
    B = np.full((L, L, len(bins)), 0)
    for i in range(L):
        for j in range(L):
            for bin_i, bin_range in bins.items():
                min_max = [float(x) for x in bin_range.split()]
                if Y[i, j] > min_max[0] and Y[i, j] <= min_max[1]:
                    B[i, j, bin_i] = 1
    return B


def slide_and_cut(X, Y, window_size, stride, output_pid=False, datatype=4):
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]
        if tmp_Y == 0:
            i_stride = stride
        elif tmp_Y == 1:
            if datatype == 4:
                i_stride = stride//6
            elif datatype == 2:
                i_stride = stride//10
            elif datatype == 2.1:
                i_stride = stride//7
        elif tmp_Y == 2:
            i_stride = stride//2
        elif tmp_Y == 3:
            i_stride = stride//20
        for j in range(0, len(tmp_ts)-window_size, i_stride):
            out_X.append(tmp_ts[j:j+window_size])
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
