from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pfbayes.common.torch_utils import MLP
from pfbayes.common.pytorch_initializer import weights_init
from pfbayes.common.consts import NONLINEARITIES


class HyperNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, nonlinearity):
        super(HyperNet, self).__init__()
        self.mlp = MLP(input_dim, hidden_dims, nonlinearity, act_last=nonlinearity)
        weights_init(self)

    def forward(self, x, list_shapes):
        params = self.mlp(x).view(-1)
        list_tensors = []
        pos = 0
        for s in list_shapes:
            num_elements = np.prod(s)
            list_tensors.append(params[pos : pos + num_elements].view(s))
        return list_tensors


class OdeLayer(nn.Module):
    def __init__(self, args, dim_in, dim_out, cond_dim):
        super(OdeLayer, self).__init__()
        self.layer_cond = args.layer_cond
        self.set_param_shapes(args, dim_in, dim_out, cond_dim)

        if args.layer_cond == 'hyper':
            total_shape = sum([np.prod(s) for s in self.list_param_shapes])
            self.hyper_net = HyperNet(cond_dim, args.hyper_dims + '-' + str(total_shape), args.nonlinearity)
        else:
            self.create_param_from_shapes(self.list_param_shapes)
        weights_init(self)

    def set_param_shapes(self, args):
        raise NotImplementedError

    def create_param_from_shapes(self, list_param_shapes):
        self.list_params = []
        for s in list_param_shapes:
            self.list_params.append(Parameter(torch.Tensor(s)))
        self.list_params = nn.ParameterList(self.list_params)

    def get_params(self, cond_tensor):
        if self.layer_cond == 'hyper':
            return self.hyper_net(cond_tensor, self.list_param_shapes)
        else:
            return self.list_params


class ConcatSquashLinear(OdeLayer):
    def __init__(self, args, dim_in, dim_out, cond_dim):
        super(ConcatSquashLinear, self).__init__(args, dim_in, dim_out, cond_dim)

    def set_param_shapes(self, args, dim_in, dim_out, cond_dim):
        input_dim = dim_in
        if self.layer_cond == 'concat':
            input_dim += cond_dim
        self.list_param_shapes = [torch.Size([dim_out, input_dim]),  # self._layer_w
                                  torch.Size([dim_out]),  # self._layer_b
                                  torch.Size([dim_out, 1]),  # self._hyper_bias
                                  torch.Size([dim_out, 1]),  # self._hyper_gate_w
                                  torch.Size([dim_out])]  # self._hyper_gate_b

    def forward(self, t, x, context=None):
        if self.layer_cond == 'hyper':
            cond_tensor = context
        else:
            cond_tensor = None
        layer_w, layer_b, hyper_bias, hyper_gate_w, hyper_gate_b = self.get_params(cond_tensor)

        if self.layer_cond == 'concat':            
            x = torch.cat([x, context], dim=-1)

        h1 = F.linear(x, layer_w, layer_b)
        gate = F.linear(t.view(1, 1), hyper_gate_w, hyper_gate_b)
        b = t.view(1, 1) * hyper_gate_b

        return h1 * torch.sigmoid(gate) + b


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1))


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, 
                prior_samples=None, ob_m=None, 
                reverse=False, inds=None, integration_times=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, prior_samples=prior_samples, ob_m=ob_m, 
                                  reverse=reverse, integration_times=integration_times)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, prior_samples=prior_samples, ob_m=ob_m,
                                         reverse=reverse, integration_times=integration_times)
            return x, logpx
