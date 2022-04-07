import pdb
from copy import deepcopy
from itertools import product
from operator import itemgetter
import torch
from torch import nn, optim
from torch.nn import functional as F
from relax.nas import Supernet, get_module
from relax.ops import Conv, Fourier
from relax.xd import Pad, Unpad, int2tuple, pad2size


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (torch.Tensor, float, bool, float, int) -> torch.Tensor

    def _gen_gumbels():
      gumbels = -torch.empty_like(logits).cuda().exponential_().log()
      if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
        # to avoid zero in exp output
        gumbels = _gen_gumbels()
      return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
      # Straight through.
      index = y_soft.max(dim, keepdim=True)[1]
      y_hard = torch.zeros_like(logits).cuda().scatter_(dim, index, 1.0)
      ret = y_hard - y_soft.detach() + y_soft
    else:
      # Reparametrization trick.
      ret = y_soft

    if torch.isnan(ret).sum():
      raise OverflowError(f'gumbel softmax output: {ret}')
    return ret

def get_weight(temp, w):
    return gumbel_softmax(F.log_softmax(w, dim=-1), tau=temp, hard=False, dim=-1)


class MultiConvPerm(nn.Module):

    def __init__(self, kernel_sizes, dilations, **kwargs):

        super(MultiConvPerm, self).__init__()

        self.dim = len(kernel_sizes[0])
        self.kd = list(product(kernel_sizes, dilations))
        self.shape = elementwise_max([[d*(k-1)+1 for k, d in zip(ks, ds)] for ks, ds in self.kd])
        self.max_l = self.shape[0]
        self.ks, self.ds = [k[0] for k in kernel_sizes], [d[0] for d in dilations]
        self.max_k = max(self.ks)

        self.temp = kwargs['temp']
        self.stream = kwargs['stream']
        self.device = torch.device(kwargs['device'])

        self.alpha = nn.Parameter(1e-3 * torch.randn(len(self.kd)))
            

    def forward(self, weight):

        coef = get_weight(self.temp, self.alpha)

        w, input_size, kwargs = weight 
        shape = w[0].shape[:2] + self.shape
        if not self.stream:
            cat = torch.empty(*shape, len(self.kd), device=self.device)

        for i, (ks, ds) in enumerate(self.kd):
            weight = w[i]

            for j, (k, d) in enumerate(zip(ks, ds)):
                if d > 1:
                    p = torch.zeros((1, 1) + tuple([d] * self.dim), device=self.device)
                    p[tuple([0] * (self.dim + 2))] = 1
                    if self.dim == 2:
                        weight = torch.kron(weight.transpose(-1, j+2).contiguous(),p)[...,:-(d-1),:-(d-1)].transpose(-1, j+2)
                    else:
                        weight = torch.kron(weight.transpose(-1, j+2).contiguous(),p)[...,:-(d-1)].transpose(-1, j+2)
                break

            weight = F.pad(weight, sum((((s-d*(k-1)-1)//2, (s-d*(k-1))//2) for k, d, s in zip(ks, ds, shape[2:])), ()))
            if self.stream:
                if i == 0:
                    w_ = coef[i] * weight
                else:
                    w_ += coef[i] * weight
            else:
                cat[...,i] = weight
            
        w = F.linear(cat, coef) if not self.stream else w_

        w = pad2size(w, input_size, **kwargs)
        for i, k in enumerate(reversed(self.shape)):
            w = w.transpose(-1, -i-1).flip(-1).roll((k+1)//2, dims=-1).transpose(-1, -i-1)

        return w


def elementwise_max(entries):

    return tuple(max(e[i] for e in entries) for i in range(len(entries[0])))


def mixture(kernel_sizes, *args, dilations=[1], stride=1, padding=0, padding_mode='zeros', compact=True, odd=False, **kwargs):

    dims = len(kernel_sizes[0])
    max_kernel_size = elementwise_max(kernel_sizes)
    max_dilation = elementwise_max(dilations)
    padding = [max_dilation[0] * (max_kernel_size[0] - 1) // 2] * len(max_kernel_size)
    inpad = Pad(dims, padding=tuple(padding), padding_mode=padding_mode, dilation=max_dilation, kernel_size=max_kernel_size)
    K = Fourier(inv=True, normalized=True, dims=dims, compact=compact, odd=odd)
    L = nn.Sequential(MultiConvPerm(kernel_sizes, dilations, **kwargs), Fourier(dims=dims, compact=compact))
    M = Fourier(normalized=True, dims=dims, compact=compact)
    unpad = Unpad(dims, stride=stride)
    return K, L, M, inpad, unpad


class MixtureSupernet(Supernet):

    def conv2mixture(self, sample_input, *args, named_modules=None, kernel_sizes=None, dilations=None, dims=1, **kwargs):
        shape_idx = -2 if len(sample_input.size()) == 4 and sample_input.size()[-1] != sample_input.size()[-2] else -1
        field_sizes = [ d * (k-1)+1 for k in kernel_sizes for d in dilations]
        if kwargs['remain_shape']:
            field_sizes = [ 10 * k for k in kernel_sizes for d in dilations]
        
        named_modules = self.named_modules() if named_modules is None else named_modules
        named_modules = [(n, m) for n, m in named_modules if hasattr(m, 'kernel_size') and type(m.kernel_size) == tuple and type(m) == Conv(len(m.kernel_size))]
        module_io = self.collect_io(sample_input, (m for _, m in named_modules), *args)
        kernel_sizes = None if kernel_sizes is None else [int2tuple(k, length=dims, allow_other=True) for k in kernel_sizes]
        dilations = None if dilations is None else [int2tuple(d, length=dims, allow_other=True) for d in dilations]
        max_kernel_size = None if kernel_sizes is None else elementwise_max(kernel_sizes)
        max_dilation = None if dilations is None else elementwise_max(dilations)

        for i, (name, module) in enumerate(named_modules):
            if 'downsample' in name or 'pool' in name or 'shortcut' in name or 'channel_match' in name:
                continue
            ks = [module.kernel_size] if kernel_sizes is None else kernel_sizes
            ds = [module.dilation] if dilations is None else dilations
            module_input, module_output = module_io.get(module, (None, None))
            separable = module.out_channels == module.in_channels and kwargs['separable']
            self.patch2xd(name,
                          module_input, module_output,
                          ks,
                          *args,
                          dilations=ds,
                          padding=[p - (d*(k-1)//2) + ((d if dilations is None else max_dilation[i]) * ((k if kernel_sizes is None else max_kernel_size[i])-1) // 2) for i, (k, d, p) in enumerate(zip(module.kernel_size, module.dilation, module.padding))],
                          padding_mode=module.padding_mode,
                          stride=module.stride,
                          groups=module.out_channels if separable else module.groups,
                          weight=nn.ParameterList(nn.Parameter(torch.normal(0, 0.01, (module.weight.shape[0], 1 if separable else module.weight.shape[1], *k))) for k in ks for d in ds),
                          bias=None if module.bias is None else nn.Parameter(module.bias.data),
                          arch=mixture,
                          **kwargs)

        self.grad_scale = kwargs['grad_scale']
        self.field_sizes = torch.tensor(field_sizes, device=kwargs['device']).float()


    def remove_module(self, name, named_modules=None):

        named_modules = self.named_modules() if named_modules is None else named_modules
        mods = [(n, m) for n, m in named_modules if name in n]

        for name, module in mods:
            module.remove = True


    def scale_grad(self):
        if self.grad_scale:
            for j, (name, param) in enumerate(self.named_arch_params()):
                param.grad /= self.grad_scale * (param.grad.norm() + 1e-8)
                param.grad += 1e-5 * self.field_sizes
        else:
            for j, (name, param) in enumerate(self.named_arch_params()):
                param.grad += 1e-5 * self.field_sizes
