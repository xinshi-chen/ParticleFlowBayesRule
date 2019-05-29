from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.distributions.multinomial import Multinomial

import numpy as np
import math
from scipy.stats import multivariate_normal, gaussian_kde

from pfbayes.common.consts import DEVICE, t_float
from pfbayes.common.torch_utils import log_sum_exp
from pfbayes.common.torch_utils import pairwise_distances
from pfbayes.common.pytorch_initializer import glorot_uniform
import collections

class MyMulti(torch.distributions.Multinomial):
    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        samples = self._categorical.sample(torch.Size((self.total_count,)) + sample_shape)
        # samples.shape is (total_count, sample_shape, batch_shape), need to change it to
        # (sample_shape, batch_shape, total_count)
        shifted_idx = list(range(samples.dim()))
        shifted_idx.append(shifted_idx.pop(0))
        samples = samples.permute(*shifted_idx)
        return samples

def torch_randn2d(n_rows, n_cols):
    return torch.tensor(np.random.randn(n_rows, n_cols), dtype=t_float).to(DEVICE)


def mix_gauss_pdf(x, landmarks, bw):
    assert x.shape[1] == landmarks.shape[1]
    num = landmarks.shape[0]
    dim = landmarks.shape[1]
    dist = pairwise_distances(x.to(DEVICE) / bw, landmarks.to(DEVICE) / bw)
    log_comp = -0.5 * dist - 0.5 * np.log(2 * np.pi) * dim - np.log(num)
    if isinstance(bw, float):  # scalar
        log_comp = log_comp - dim * np.log(bw)
    else:
        assert bw.shape[1] == dim
        log_comp = log_comp - torch.sum(torch.log(bw))
    return log_sum_exp(log_comp, keepdim=True)


def get_gamma(X, bandwidth):
    with torch.no_grad():
        x_norm = torch.sum(X ** 2, dim=1, keepdim=True)
        x_t = torch.transpose(X, 0, 1)
        x_norm_t = x_norm.view(1, -1)
        t = x_norm + x_norm_t - 2.0 * torch.matmul(X, x_t)
        dist2 = F.relu(Variable(t)).detach().data

        d = dist2.cpu().numpy()
        d = d[np.isfinite(d)]
        d = d[d > 0]
        median_dist2 = float(np.median(d))
        gamma = 0.5 / median_dist2 / bandwidth
        return gamma


def get_kernel_mat(x, landmarks, gamma):
    d = pairwise_distances(x, landmarks)
    k = torch.exp(d * -gamma)
    k = k.view(x.shape[0], -1)
    return k


def MMD(x, y, bandwidth=1.0):
    y = y.detach()
    gamma = get_gamma(y.detach(), bandwidth)
    kxx = get_kernel_mat(x, x, gamma)
    idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
    kxx = kxx * (1 - torch.eye(x.shape[0]).to(DEVICE))
    kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

    kyy = get_kernel_mat(y, y, gamma)
    idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
    kyy[idx, idx] = 0.0
    kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
    kxy = torch.sum(get_kernel_mat(y, x, gamma)) / x.shape[0] / y.shape[0]
    mmd = kxx + kyy - 2 * kxy
    return mmd


class KDE(object):
    def __init__(self, landmarks, coeff=1.0):
        self.landmarks = landmarks
        with torch.no_grad():
            n = landmarks.shape[0]
            self.num = n
            s = torch.std(landmarks, dim=0, keepdim=True)
            # Silverman's rule of thumb 
            self.dim = landmarks.shape[1]
            self.h = np.power(4.0 / (self.dim + 2.0), 1.0 / (self.dim + 4))
            self.h *= np.power(self.landmarks.shape[0], -1.0 / (self.dim + 4))        
            self.h = self.h * s * coeff
            self.h = self.h.to(DEVICE)
        self.landmarks = self.landmarks.to(DEVICE)
        num_landmarks = self.landmarks.shape[0]
        p = torch.ones(num_landmarks, dtype=t_float) / float(num_landmarks)
        self.idx_sampler = Multinomial(probs=p)

    def log_pdf(self, x):
        return mix_gauss_pdf(x, self.landmarks, self.h)

    def get_samples(self, num_samples):
        idx = self.idx_sampler.sample(sample_shape=[num_samples]).to(DEVICE)
        centers = torch.matmul(idx, self.landmarks)

        z = torch_randn2d(num_samples, self.dim) * self.h + centers
        return z


class GammaDist(object):
    def __init__(self, alpha, beta):
        '''
        Args:
            alpha: shape parameter
            beta: rate parameter
        '''
        self.alpha = alpha
        self.beta = beta

    def log_pdf(self, log_x):
        return (self.alpha - 1) * log_x - self.beta * torch.exp(log_x) + self.alpha * np.log(self.beta) - np.log(math.gamma(self.alpha))

    def get_log_pdf(self, log_x):
        return self.log_pdf(log_x)

    def get_samples(self, num_samples):
        rand_vars = np.random.gamma(self.alpha, 1.0 / self.beta, size=[num_samples, 1])  # numpy uses scale parameter
        return torch.tensor(np.log(rand_vars), 
                            dtype=t_float).to(DEVICE)


class BayesianNNPrior(object):
    def __init__(self, alpha, beta, input_dim, hidden_size):
        self.gamma_dist = GammaDist(alpha, beta)
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.param_sizes = [self.input_dim * self.hidden_size, self.hidden_size, self.hidden_size, 1, 1, 1]
        self.num_params = np.sum(self.param_sizes)
        self.num_nn_params = self.num_params - 2

    def get_samples(self, num_samples):
        w1 = torch.Tensor(num_samples, self.input_dim * self.hidden_size).to(DEVICE)        
        b1 = torch.zeros(num_samples, self.hidden_size, dtype=t_float).to(DEVICE)
        w2 = torch.Tensor(num_samples, self.hidden_size).to(DEVICE)
        b2 = torch.zeros(num_samples, 1, dtype=t_float).to(DEVICE)
        glorot_uniform(w1)
        glorot_uniform(w2)
        loggamma = self.gamma_dist.get_samples(num_samples).to(DEVICE)
        loglambda = self.gamma_dist.get_samples(num_samples).to(DEVICE)
        cat_param = torch.cat((w1, b1, w2, b2, loggamma, loglambda), dim=-1)
        return cat_param

    def log_pdf(self, particles):
        if isinstance(particles, tuple) or isinstance(particles, list):
            w1, b1, w2, b2, loggamma, loglambda = particles
            nn_params = torch.cat([w1, b1, w2, b2], dim=-1)
        else:
            assert len(particles.shape) == 2
            nn_params, loggamma, loglambda = torch.split(particles, [self.num_nn_params, 1, 1], dim=1)

        # gamma prior
        ll_hyper = self.gamma_dist.log_pdf(loglambda) + self.gamma_dist.log_pdf(loggamma)
        # gaussian prior
        ll_w = -nn_params ** 2 / 2.0 * torch.exp(loglambda) - 0.5 * (np.log(2 * np.pi) - loglambda)
        ll = ll_hyper + torch.sum(ll_w, dim=-1, keepdim=True)
        return ll

    def get_log_pdf(self, particles):
        return self.log_pdf(particles)


class DiagGaussianDist(nn.Module):
    def __init__(self, dim, mu=0.0, logvar=0.0):
        super(DiagGaussianDist, self).__init__()
        self.dim = dim
        self.mu = Parameter(torch.zeros(1, dim, dtype=t_float) + mu)
        self.logvar = Parameter(torch.zeros(1, dim, dtype=t_float) + logvar)

    def forward(self, num_samples):  # sampling
        std = torch.exp(self.logvar * 0.5)        
        return torch_randn2d(num_samples, self.dim) * std + self.mu


class DiagMvn(object):
    def __init__(self, mu, sigma):
        if isinstance(mu, np.ndarray) or isinstance(mu, list):
            mu = np.array(mu, dtype=np.float32).reshape(1, -1)
            mu = torch.tensor(mu)
            sigma = np.array(sigma, dtype=np.float32).reshape(1, -1)
            sigma = torch.tensor(sigma)
        mu = mu.to(DEVICE)
        sigma = sigma.to(DEVICE)
        assert mu.shape[1] == sigma.shape[1]
        self.mu = mu
        self.sigma = sigma        
        self.gauss_dim = mu.shape[1]

    def get_log_pdf(self, x):
        return DiagMvn.log_pdf(x, self.mu, self.sigma)

    def get_samples(self, num_samples):
        return torch_randn2d(num_samples, self.gauss_dim) * self.sigma + self.mu

    @staticmethod
    def log_pdf(x, mu=None, sigma=None):
        if mu is None:
            mu = 0.0
        t = -(x - mu) ** 2 / 2.0
        if sigma is not None:
            t = t / sigma / sigma
            t = t - 0.5 * torch.log(2 * np.pi * sigma * sigma)
        else:
            t = t - 0.5 * np.log(2 * np.pi)
        return torch.sum(t, dim=-1, keepdim=True)

    @staticmethod
    def grad_mu_log_pdf(x, mu=None, sigma=None):
        if mu is None:
            mu = 0.0
        if not isinstance(x, collections.Sized):
            num_x = 1
            sum_x = x
        else:
            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x)
            if x.dim() == 1:
                num_x = 1
                sum_x = x
            else:
                num_x = x.shape[0]
                sum_x = torch.sum(x, dim=0, keepdim=True)
        t = sum_x - num_x * mu
        if sigma is not None:
            t = t / sigma / sigma
        # return tensor of the same shape as mu
        return t

    @staticmethod
    def grad_x_log_pdf(x, mu=None, sigma=None):
        if mu is None:
            mu = 0.0
        t = mu - x
        if sigma is not None:
            t = t / sigma / sigma
        return t


if __name__ == '__main__':
    g = GammaDist(1, 10)
    s = g.get_samples(10)
    print(torch.exp(s))
    print(g.log_pdf(s))
