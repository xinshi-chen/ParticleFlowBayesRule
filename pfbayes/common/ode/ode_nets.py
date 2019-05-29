from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import torch
import torch.nn as nn
from pfbayes.common.torch_utils import MLP
from pfbayes.common.consts import NONLINEARITIES
import pfbayes.common.ode.ode_layers as ode_layers
from pfbayes.common.pytorch_initializer import weights_init


class KernelEmbedNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, nonlinearity, trainable=False):
        super(KernelEmbedNet, self).__init__()
        self.mlp = MLP(input_dim, hidden_dims, nonlinearity, act_last=nonlinearity)
        self.output_size = self.mlp.output_size
        self.trainable = trainable

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)
        with torch.set_grad_enabled(self.trainable):
            phi_x = self.mlp(x)
            return torch.mean(phi_x, dim=0, keepdim=True)

class SizeBalancedEmbedNet(KernelEmbedNet):
    def __init__(self, x_dim, y_dim, hidden_dims, nonlinearity, trainable=False):
        input_dim = max(x_dim, y_dim) * 2        
        super(SizeBalancedEmbedNet, self).__init__(input_dim, hidden_dims, nonlinearity, trainable)
        self.x_dim = x_dim
        self.y_dim = y_dim
        if x_dim > y_dim:
            self.mm = nn.Linear(y_dim, x_dim)
        else:
            self.mm = nn.Linear(x_dim, y_dim)
        weights_init(self)

    def forward(self, t):
        assert isinstance(t, tuple)
        x, y = t
        if self.x_dim > self.y_dim:
            y = self.mm(y)
        else:
            x = self.mm(x)
        x = torch.cat([x, y], dim=1)
        with torch.set_grad_enabled(self.trainable):
            phi_x = self.mlp(x)
            return torch.mean(phi_x, dim=0, keepdim=True)


class TimePredNet(nn.Module):
    def __init__(self, prior_embed_dim, ob_dim, hidden_dims, nonlinearity, 
                 time_length=1.0):
        super(TimePredNet, self).__init__()
        self.time_length = time_length
        self.ob2embed = nn.Linear(ob_dim, prior_embed_dim)
        self.st_pred = MLP(input_dim=prior_embed_dim * 2,
                           hidden_dims=hidden_dims,
                           nonlinearity=nonlinearity,
                           act_last='softplus')
        self.dur_pred = MLP(input_dim=prior_embed_dim * 2,
                            hidden_dims=hidden_dims,
                            nonlinearity=nonlinearity,
                            act_last='sigmoid')
        weights_init(self)

    def forward(self, prior_embed, ob_m):
        ob_embed = self.ob2embed(ob_m)
        cur_input = torch.cat((prior_embed, ob_embed), dim=-1)
        st_pred = torch.squeeze(self.st_pred(cur_input))
        ed_pred = st_pred + torch.squeeze(self.dur_pred(cur_input)) * self.time_length
        return torch.stack((st_pred, ed_pred))


class ContextCondNet(nn.Module):
    def __init__(self, args, x_dim, prior_embed_dim, ob_dim, ob_trans_func):
        super(ContextCondNet, self).__init__()
        self.output_size = prior_embed_dim
        base_layer = getattr(ode_layers, args.layer_type)
        if args.ob_feed == 'ob':
            self.ob2embed = nn.Linear(ob_dim, prior_embed_dim)
        elif args.ob_feed == 'grad_x':
            self.ob2embed = nn.Linear(x_dim, prior_embed_dim)
        else:
            raise NotImplementedError
        self.ob_trans_func = ob_trans_func
        self.x2embed = nn.Linear(x_dim, prior_embed_dim)
        self.linear_merge = nn.Linear(prior_embed_dim * 3, prior_embed_dim)
        self.act_func = NONLINEARITIES['tanh']        

    def forward(self, t, x, prior_embed=None, ob_m=None):
        ob_m = self.ob_trans_func(ob_m, x)
        embed_ob = self.ob2embed(ob_m)
        embed_x = self.x2embed(x)
        if prior_embed.shape[0] == 1:
            prior_embed = prior_embed.repeat(x.shape[0], 1)
        if embed_ob.shape[0] == 1:
            embed_ob = embed_ob.repeat(x.shape[0], 1)
        merged = torch.cat([embed_x, embed_ob, prior_embed], dim=-1)
        h = self.act_func(self.linear_merge(merged))
        return h


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(self, args, x_dim, context_cond_net=None, prior_embed_dim=0, ob_dim=0):
        super(ODEnet, self).__init__()
        hidden_dims = tuple(map(int, args.dims.split("-")))
        layer_type=args.layer_type
        nonlinearity=args.nonlinearity
        base_layer = getattr(ode_layers, layer_type)
        
        self.context_cond_net = context_cond_net
        self.layer_cond = args.layer_cond
        # build layers and add them
        layers = []
        activation_fns = []
        if self.context_cond_net is not None:
            prev_size = self.context_cond_net.output_size
        else:
            prev_size = x_dim
        if prior_embed_dim is None:
            prior_embed_dim = 0
        if ob_dim is None:
            ob_dim = 0

        for dim_out in hidden_dims + (x_dim,):
            layer = base_layer(args, prev_size, dim_out, prior_embed_dim + ob_dim)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            prev_size = dim_out

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y, prior_embed=None, ob_m=None):
        dx = y
        if self.context_cond_net is not None:
            dx = self.context_cond_net(t, y, prior_embed, ob_m)
        if self.layer_cond != 'ignore':
            context = torch.cat([prior_embed, ob_m], dim=-1).repeat(y.shape[0], 1)
        else:
            context = None
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx, context)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx
