from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import copy
import numpy as np
import torch
import torch.nn as nn
import pfbayes.common.ode.ode_layers as ode_layers
from pfbayes.common.consts import NONLINEARITIES, DEVICE
from pfbayes.common.cmd_args import cmd_args


def divergence_bf(dx, y, **unused_kwargs):
    if cmd_args.phase == 'eval_metric':
        return torch.zeros(dx.shape[0]).to(DEVICE)
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


class ODEfunc(nn.Module):

    def __init__(self, diffeq, divergence_fn="brute_force", residual=False, rademacher=False):
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        self.diffeq = diffeq
        assert residual == False  # not sure what it is for, disable for now
        self.residual = residual
        self.rademacher = rademacher

        assert divergence_fn == "brute_force"
        self.divergence_fn = divergence_bf

        self.prior_embed = None  # embedding of prior distribution
        self.ob_m = None  # latest observation
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def set_context(self, prior_embed=None, ob_m=None):
        self.prior_embed = prior_embed
        self.ob_m = ob_m
        
    def reset_context(self):
        self.prior_embed = None
        self.ob_m = None

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = sample_rademacher_like(y)
            else:
                self._e = sample_gaussian_like(y)
        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, prior_embed=self.prior_embed, ob_m=self.ob_m, *states[2:])
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)   
        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * torch.tensor(np.prod(y.shape[1:]), dtype=torch.float32
                                                                     ).to(divergence)
        return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])
