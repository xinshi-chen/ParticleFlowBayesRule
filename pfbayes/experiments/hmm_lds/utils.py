from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import torch
import numpy as np
from pfbayes.common.data_utils.dataset import ToyDataset
from pfbayes.common.distributions import DiagMvn, mix_gauss_pdf, torch_randn2d, MyMulti
from pfbayes.common.consts import DEVICE
from pfbayes.common.cmd_args import cmd_args
import sklearn.metrics.pairwise as sk_metric
import random
import os
import pickle
from scipy import stats


class HmmLdsDataset(object):

    def __init__(self, dim,
                 prior_mu=0.0,
                 prior_sigma=1.0,
                 t_sigma=1.0,
                 l_sigma=1.0,
                 t_aa=None,  # transition model, matrix A
                 l_bb=None,  # likelihood model, matrix B
                 partition_sizes={}):
        self.dim = dim
        self.prior_mu = torch.tensor([[prior_mu] * dim], dtype=torch.float32).to(DEVICE)
        self.prior_sigma = torch.tensor([[prior_sigma] * dim], dtype=torch.float32).to(DEVICE)
        self.l_sigma = torch.tensor([[l_sigma] * dim], dtype=torch.float32).to(DEVICE)
        self.t_sigma = torch.tensor([[t_sigma] * dim], dtype=torch.float32).to(DEVICE)
        self._set_model()
        # super(HmmLdsDataset, self).__init__(partition_sizes=partition_sizes)

    def _set_model(self):
        full_path = os.path.realpath(__file__)
        path = os.path.dirname(full_path)
        filename = path+'/data/lds_model%d.pkl' % cmd_args.gauss_dim

        with open(filename, 'rb') as file:
            matrix_a, matrix_b = pickle.load(file)

        self.t_aa = matrix_a.to(DEVICE)
        self.l_bb = matrix_b.to(DEVICE)

    def gen_seq_obs(self, num_obs):
        mu = torch_randn2d(1, self.dim) * self.prior_sigma + self.prior_mu
        for i in range(num_obs):
            mu = torch.mm(mu, self.t_aa.t()) + torch_randn2d(1, self.dim) * self.t_sigma
            ob = torch.mm(mu, self.l_bb.t()) + torch_randn2d(1, self.dim) * self.l_sigma
            yield ob.detach()

    def gen_batch_obs(self, num_obs):
        pass

    def gen_seq_x_and_obs(self, num_obs):
        mu = torch_randn2d(1, self.dim) * self.prior_sigma + self.prior_mu        
        for i in range(num_obs):
            mu = torch.mm(mu, self.t_aa.t()) + torch_randn2d(1, self.dim) * self.t_sigma
            ob = torch.mm(mu, self.l_bb.t()) + torch_randn2d(1, self.dim) * self.l_sigma
            yield mu.detach(), ob.detach()

    def transition_sample(self, x):
        num_samples = x.shape[0]
        return torch.mm(x, self.t_aa.t()) + torch_randn2d(num_samples, self.dim) * self.t_sigma

    def transition_log_pdf(self, x_new, x_old):
        mu = torch.mm(x_old, self.t_aa.t())
        return mix_gauss_pdf(x_new, mu, self.t_sigma)

    def log_likelihood(self, obs, mu):
        assert mu.shape[0] == 1 or obs.shape[0] == 1  # safe bradcasting
        mu_bb = torch.mm(mu, self.l_bb.t())
        return DiagMvn.log_pdf(obs, mu_bb, self.l_sigma)

    def get_true_new_posterior(self, obs, current_mu, current_cov):
        assert obs.shape[0] == 1
        l_cov = self.l_sigma * self.l_sigma
        t_cov = self.t_sigma * self.t_sigma

        b_lcov_inv = self.l_bb.t() / l_cov
        b_lcov_b = torch.mm(self.l_bb.t(), l_cov.t() * self.l_bb)
        t_cov = torch.diag(t_cov.reshape(self.dim))

        # p(x_t | o_{1:t-1}) ~ N(trans_mu_t, trans_sigma_t)
        trans_mu_t = torch.mm(current_mu, self.t_aa.t())
        trans_cov_t = torch.mm(torch.mm(self.t_aa, current_cov), self.t_aa.t()) + t_cov

        pred_ob = torch.mm(trans_mu_t, self.l_bb.t())
        residual = obs - pred_ob
        matrix_k = torch.mm((trans_cov_t.inverse() + b_lcov_b).inverse(), b_lcov_inv)

        # p(x_t | o_{1:t}) ~ N(mu_t, sigma_t)
        mu_t = trans_mu_t + torch.mm(residual, matrix_k.t())
        cov_t = torch.mm(torch.eye(self.dim, dtype=torch.float).to(DEVICE)-torch.mm(matrix_k, self.l_bb), trans_cov_t)

        return mu_t, cov_t


class HmmLdsTestset(HmmLdsDataset):
    def __init__(self):

        fname = './data/obs_test_dim%d_len%d.npy' % (cmd_args.gauss_dim, cmd_args.test_length)
        obs = np.load(fname)
        self.obs = obs
        self.epoch = obs.shape[0]
        self.len_seq = obs.shape[1]
        super(HmmLdsTestset, self).__init__(dim=obs.shape[2])