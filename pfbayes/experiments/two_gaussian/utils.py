from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import torch
import numpy as np
from pfbayes.common.torch_utils import log_sum_exp
from pfbayes.common.data_utils.dataset import ToyDataset
from pfbayes.common.distributions import DiagMvn, torch_randn2d
from pfbayes.common.consts import DEVICE
import torch.nn.functional as F


class TwoGaussDataset(ToyDataset):

    def __init__(self,
                 prior_mu=0.0, 
                 prior_sigma=1.0, 
                 l_sigma=1.0,
                 p=0.5,
                 mu_given=None,
                 partition_sizes={}):
        self.dim = 1
        if isinstance(prior_mu, list):
            self.prior_mu = torch.tensor(np.array(prior_mu), dtype=torch.float32).view(1, -1).to(DEVICE)
        else:
            self.prior_mu = torch.tensor([[prior_mu] * 2], dtype=torch.float32).to(DEVICE)
        self.prior_sigma = torch.tensor([[prior_sigma] * 2], dtype=torch.float32).to(DEVICE)
        self.l_sigma = torch.tensor([[l_sigma]], dtype=torch.float32).to(DEVICE)
        self.p = p
        self._reset(mu_given)
        super(TwoGaussDataset, self).__init__(partition_sizes=partition_sizes)

    def _reset(self, mu_given=None):
        if mu_given is not None:
            self.latent_mu = torch.Tensor(np.array(mu_given, dtype=np.float32).reshape(1, 2)).to(DEVICE)
        else:
            self.latent_mu = torch_randn2d(1, 2) * self.prior_sigma + self.prior_mu

    def gen_batch_obs(self, num_obs):
        mu = self.latent_mu
        num_pos = np.random.binomial(num_obs, self.p)
        num_neg = num_obs - num_pos

        pos_obs = torch_randn2d(num_obs, 1) * self.l_sigma + mu[:, 0].view(-1, 1)
        neg_obs = torch_randn2d(num_obs, 1) * self.l_sigma + (mu[:, 0] + mu[:, 1]).view(-1, 1)

        perms = torch.randperm(num_obs)
        pos_indices = perms[0 : num_pos]
        neg_indices = perms[num_pos : ]

        obs = torch.cat([pos_obs[pos_indices, :], neg_obs[neg_indices, :]], dim=0)
        obs = obs[perms, :]
        return obs

    def log_prior(self, mu):
        return DiagMvn.log_pdf(mu, self.prior_mu, self.prior_sigma)

    def log_likelihood(self, obs, mu):
        assert mu.shape[0] == 1 or obs.shape[0] == 1  # safe bradcasting

        pos_ll = DiagMvn.log_pdf(obs, mu[:, 0].view(-1, 1), self.l_sigma) + np.log(self.p)
        neg_ll = DiagMvn.log_pdf(obs, torch.sum(mu, dim=1).view(-1, 1), self.l_sigma) + np.log(1.0 - self.p)
        logits = torch.cat([pos_ll, neg_ll], dim=1)
        ll = log_sum_exp(logits)
        return ll

    def get_true_posterior(self, obs):
        raise NotImplementedError
        
    def log_posterior(self, mu, obs):
        raise NotImplementedError
