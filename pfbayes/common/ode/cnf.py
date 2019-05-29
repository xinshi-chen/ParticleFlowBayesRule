from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import torch
import torch.nn as nn

from pfbayes.common.consts import t_float, cond_odeint


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, time_mode='fixed', time_pred_net=None,
                 kernel_embedding_net=None, regularization_fns=None, fwd_solver='dopri5', backwd_solver=None, atol=1e-5, rtol=1e-5):
        super(CNF, self).__init__()

        self.init_duration = T
        # how to represent the duration of integration
        self.time_mode = time_mode
        if time_mode == 'param':
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        elif time_mode == 'fixed':
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))
        else:
            assert time_mode == 'adaptive'
            self.time_pred_net = time_pred_net

        # regularization
        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)

        self.kernel_embedding_net = kernel_embedding_net
        self.odefunc = odefunc
        self.nreg = nreg
        self.regularization_states = None
        self.fwd_solver = fwd_solver
        if backwd_solver is None:
            self.backwd_solver = fwd_solver
        else:
            self.backwd_solver = fwd_solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = fwd_solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def forward(self, z, logpz=None, prior_samples=None, ob_m=None, integration_times=None, reverse=False):

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        prior_embed = None
        if prior_samples is not None:
            prior_embed = self.kernel_embedding_net(prior_samples.detach())
        if integration_times is None:
            if self.time_mode == 'adaptive':
                assert prior_embed is not None and ob_m is not None
                integration_times = self.time_pred_net(prior_embed, ob_m)
            else:
                integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)
        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.
        reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.nreg))

        if self.training:
            state_t = cond_odeint(
                self.odefunc,
                (z, _logpz) + reg_states,
                integration_times.to(z),
                context={'prior_embed': prior_embed, 'ob_m': ob_m},
                atol=[self.atol, self.atol] + [1e20] * len(reg_states) if self.fwd_solver == 'dopri5' else self.atol,
                rtol=[self.rtol, self.rtol] + [1e20] * len(reg_states) if self.fwd_solver == 'dopri5' else self.rtol,
                fwd_method=self.fwd_solver,
                backwd_method=self.backwd_solver,
                options=self.solver_options,
            )
        else:
            state_t = cond_odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                context={'prior_embed': prior_embed, 'ob_m': ob_m},
                atol=self.test_atol,
                rtol=self.test_rtol,
                fwd_method=self.test_solver,
                backwd_method=self.test_solver,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        self.regularization_states = state_t[2:]

        if logpz is not None:
            return z_t, logpz_t
        else:
            return z_t


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
