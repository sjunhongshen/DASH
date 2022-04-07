import time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import operator
from itertools import product
from functools import reduce, partial
from data_loaders import load_list


"""Customized Task Metrics"""

def fnr(output, target):
    metric = maskMetric(output.squeeze().detach().cpu().numpy() > 0.5, target.squeeze().cpu().numpy())
    TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
    TPR = TP / (TP + FN)
    return 1-TPR


def tpr(output, target):
    metric = maskMetric(output.squeeze().detach().cpu().numpy() > 0.5, target.squeeze().cpu().numpy())
    TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
    TPR = TP / (TP + FN)
    return TPR


class psicov_mae(object):
    def __init__(self):
        self.pdb_list = load_list('./data/protein/psicov.lst')
        self.length_dict = {}
        for pdb in self.pdb_list:
            (ly, seqy, cb_map) = np.load('./data/protein/psicov/distance/' + pdb + '-cb.npy', allow_pickle = True)
            self.length_dict[pdb] = ly

    def __call__(self, output, target):
        targets = []
        for i in range(len(target)):
            targets.append(np.expand_dims(target[i].cpu().numpy().transpose(1, 2, 0), axis=0))
        P = output.cpu().detach().numpy().transpose(0, 2, 3, 1)

        LMAX, pad_size = 512, 10

        Y = np.full((len(targets), LMAX, LMAX, 1), np.nan)
        for i, xy in enumerate(targets):
            Y[i, :, :, 0] = xy[0, :, :, 0]
        # Average the predictions from both triangles
        for j in range(0, len(P[0, :, 0, 0])):
            for k in range(j, len(P[0, :, 0, 0])):
                P[ :, j, k, :] = (P[ :, k, j, :] + P[ :, j, k, :]) / 2.0
        P[ P < 0.01 ] = 0.01
        # Remove padding, i.e. shift up and left by int(pad_size/2)
        P[:, :LMAX-pad_size, :LMAX-pad_size, :] = P[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]
        Y[:, :LMAX-pad_size, :LMAX-pad_size, :] = Y[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]

        PRED = P
        YTRUE = Y

        mae_lr_d8_list = np.zeros(len(PRED[:, 0, 0, 0]))
        mae_mlr_d8_list = np.zeros(len(PRED[:, 0, 0, 0]))
        mae_lr_d12_list = np.zeros(len(PRED[:, 0, 0, 0]))
        mae_mlr_d12_list = np.zeros(len(PRED[:, 0, 0, 0]))
        for i in range(0, len(PRED[:, 0, 0, 0])):
            L = self.length_dict[self.pdb_list[i]]
            PAVG = np.full((L, L), 100.0)
            # Average the predictions from both triangles
            for j in range(0, L):
                for k in range(j, L):
                    PAVG[j, k] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
            # at distance 8 and separation 24
            Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
            P = np.copy(PAVG)
            for p in range(len(Y)):
                for q in range(len(Y)):
                    if q - p < 24:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
                        continue
                    if Y[p, q] > 8:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
            mae_lr_d8 = np.nan
            if not np.isnan(np.abs(Y - P)).all():
                mae_lr_d8 = np.nanmean(np.abs(Y - P))
                #mae_lr_d8 = np.sqrt(np.nanmean(np.abs(Y - P) ** 2))
            # at distance 8 and separation 12
            Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
            P = np.copy(PAVG)
            for p in range(len(Y)):
                for q in range(len(Y)):
                    if q - p < 12:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
                        continue
                    if Y[p, q] > 8:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
            mae_mlr_d8 = np.nan
            if not np.isnan(np.abs(Y - P)).all():
                mae_mlr_d8 = np.nanmean(np.abs(Y - P))
            # at distance 12 and separation 24
            Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
            P = np.copy(PAVG)
            for p in range(len(Y)):
                for q in range(len(Y)):
                    if q - p < 24:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
                        continue
                    if Y[p, q] > 12:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
            mae_lr_d12 = np.nan
            if not np.isnan(np.abs(Y - P)).all():
                mae_lr_d12 = np.nanmean(np.abs(Y - P))
            # at distance 12 and separation 12
            Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
            P = np.copy(PAVG)
            for p in range(len(Y)):
                for q in range(len(Y)):
                    if q - p < 12:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
                        continue
                    if Y[p, q] > 12:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
            mae_mlr_d12 = np.nan
            if not np.isnan(np.abs(Y - P)).all():
                mae_mlr_d12 = np.nanmean(np.abs(Y - P))
            # add to list
            mae_lr_d8_list[i] = mae_lr_d8
            mae_mlr_d8_list[i] = mae_mlr_d8
            mae_lr_d12_list[i] = mae_lr_d12
            mae_mlr_d12_list[i] = mae_mlr_d12

        mae = np.nanmean(mae_lr_d8_list)
        
        if np.isnan(mae):
            return np.inf
        return mae


def maskMetric(PD, GT):
    if len(PD.shape) == 2:
        PD = PD.reshape(1, *PD.shape)
    if len(GT.shape) == 2:
        GT = GT.reshape(1, *GT.shape)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(GT.shape[0]):
        P = GT[i].sum()
        TP += (PD[i][GT[i] == 1] == 1).sum()
        TN += (PD[i][GT[i] == 0] == 0).sum()
        FP += (PD[i][GT[i] == 0] == 1).sum()
        FN += (PD[i][GT[i] == 1] == 0).sum()
    return np.array([TP, TN, FP, FN])


def ecg_f1(output, target):
    target = target.cpu().detach().numpy()
    output = np.argmax(output.cpu().detach().numpy(), axis=1)
    tmp_report = metrics.classification_report(target, output, output_dict=True, zero_division=0)
    f1_score = []
    for i, (y, scores) in enumerate(tmp_report.items()): 
        if y == '0' or y == '1' or y == '2' or y == '3':
            f1_score.append(tmp_report[y]['f1-score'])
    f1_score = np.mean(f1_score)
    return f1_score


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    res = res[0] if len(res) == 1 else res
    return res


def accuracy_onehot(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    target = torch.argmax(target, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    res = res[0] if len(res) == 1 else res
    return res


def map_value(output, target):
    val_preds = torch.sigmoid(output).float().cpu().detach().numpy()
    val_gts = target.cpu().detach().numpy()
    map_value = metrics.average_precision_score(val_gts, val_preds, average="macro")
    return map_value


def auroc(output, target):
    output = torch.sigmoid(output).float()
    result = output.cpu().detach().numpy()
    y = target.cpu().detach().numpy()
    result_shape = np.shape(result)

    fpr_list, tpr_list, auroc_list = [], [], []
    precision_list, recall_list, aupr_list = [], [], []
    for i in range(result_shape[1]):
        fpr_temp, tpr_temp, auroc_temp  = calculate_auroc(result[:, i], y[:, i])
        precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], y[:, i])

        fpr_list.append(fpr_temp)
        tpr_list.append(tpr_temp)
        precision_list.append(precision_temp)
        recall_list.append(recall_temp)
        auroc_list.append(auroc_temp)
        aupr_list.append(aupr_temp)

    avg_auroc = np.nanmean(auroc_list)
    avg_aupr = np.nanmean(aupr_list)
    return avg_auroc


def calculate_auroc(predictions, labels):
    fpr_list, tpr_list, threshold_list = metrics.roc_curve(y_true=labels, y_score=predictions)
    score = metrics.auc(fpr_list, tpr_list)
    return fpr_list, tpr_list, score


def calculate_stats(output, target, class_indices=None):
    classes_num = target.shape[-1]
    if class_indices is None:
        class_indices = range(classes_num)
    stats = []

    for k in class_indices:
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)
        dict = {'AP': avg_precision}
        stats.append(dict)

    return stats


def calculate_aupr(predictions, labels):
    precision_list, recall_list, threshold_list = metrics.precision_recall_curve(y_true=labels, probas_pred=predictions)
    aupr = metrics.auc(recall_list, precision_list)
    return precision_list, recall_list, aupr


"""Customized Task Losses"""

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean' if self.size_average else 'sum')
        if torch.cuda.is_available():
             self.criterion =  self.criterion.cuda()

    def forward(self, output, target):
        model_out = F.softmax(output, dim = 1) + 1e-9

        ce = torch.multiply(target, -torch.log(model_out))
        weight = torch.multiply(target, (1 - model_out) ** self.gamma)
        fl = self.alpha * torch.multiply(weight, ce)
        reduced_fl = torch.sum(fl, axis=1)
        return reduced_fl.mean()


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.cuda = cuda
        self.criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='mean')


    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            
            if self.cuda:
                self.criterion = self.criterion.cuda()
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        
        loss = self.criterion(logit, target.long())
        
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n = logit.size()[0]
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


def logCoshLoss(y_t, y_prime_t, reduction='mean', eps=1e-12):
    if reduction == 'mean':
        reduce_fn = torch.mean
    elif reduction == 'sum':
        reduce_fn = torch.sum
    else:
        reduce_fn = lambda x: x
    x = y_prime_t - y_t
    return reduce_fn(torch.log((torch.exp(x) + torch.exp(-x)) / 2))


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


"""Hepler Funcs"""

def count_params(model):
    c = 0
    for p in model.parameters():
        try:
            c += reduce(operator.mul, list(p.size()))
        except:
            pass

    return c

def print_grad(model, kernel_choices, dilation_choices):
    param_values, ks, ds = [], [], []
    for name, param in model.named_arch_params():
        param_values.append(param.data.argmax(0))
        print(name, param.data)
        ks.append(kernel_choices[int(param_values[-1] // len(dilation_choices))])
        ds.append(dilation_choices[int(param_values[-1] % len(dilation_choices))])

    print("Kernel pattern (name, k, d):", ks, np.sum(ks), ds, np.sum(ds))

def mask(img, ignore):
    return img * (1 - ignore)

