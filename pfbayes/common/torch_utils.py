from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np
from pfbayes.common.consts import NONLINEARITIES
from pfbayes.common.pytorch_initializer import weights_init


def log_sum_exp(logits, keepdim=False):
    ll = torch.mean(logits - F.log_softmax(logits, dim=-1), dim=-1, keepdim=keepdim)
    return ll


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


class LogisticReg(nn.Module):
    def __init__(self, dim, num):
        super(LogisticReg, self).__init__()
        self.dim = dim
        self.w = Parameter(torch.Tensor(num, dim))
        weights_init(self)

    def forward(self, x):
        return torch.matmul(x, self.w.t())


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, nonlinearity, act_last=None):
        super(MLP, self).__init__()
        self.act_last = act_last
        hidden_dims = tuple(map(int, hidden_dims.split("-")))
        prev_size = input_dim

        layers = []
        activation_fns = []
        for h in hidden_dims:
            layers.append(nn.Linear(prev_size, h))
            prev_size = h
            activation_fns.append(NONLINEARITIES[nonlinearity])
        if act_last is not None:
            activation_fns[-1] = NONLINEARITIES[self.act_last]
        self.output_size = prev_size
        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns)
        weights_init(self)

    def forward(self, x):
        for l, layer in enumerate(self.layers):
            x = layer(x)
            if l + 1 < len(self.layers) or self.act_last is not None:
                x = self.activation_fns[l](x)
        return x