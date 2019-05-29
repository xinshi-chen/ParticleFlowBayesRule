from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import torch
import numpy as np
from pfbayes.common.data_utils.dataset import ToyDataset
from pfbayes.common.distributions import DiagMvn, torch_randn2d
from pfbayes.common.consts import DEVICE
from pfbayes.common.cmd_args import cmd_args
import sklearn.metrics.pairwise as sk_metric
import random
from pfbayes.common.distributions import KDE
from scipy import stats


class MvnUniModalDataset(ToyDataset):

    def __init__(self, dim, 
                 prior_mu=0.0, 
                 prior_sigma=1.0, 
                 l_sigma=1.0,
                 mu_given=None,
                 partition_sizes={}):        
        self.dim = dim
        self.prior_mu = torch.tensor([[prior_mu] * dim], dtype=torch.float32).to(DEVICE)
        self.prior_sigma = torch.tensor([[prior_sigma] * dim], dtype=torch.float32).to(DEVICE)
        self.l_sigma = torch.tensor([[l_sigma] * dim], dtype=torch.float32).to(DEVICE)
        self._reset(mu_given)
        super(MvnUniModalDataset, self).__init__(partition_sizes=partition_sizes)

    def _reset(self, mu_given=None):
        if mu_given is not None:
            self.latent_mu = torch.Tensor(np.array(mu_given, dtype=np.float32).reshape(1, dim)).to(DEVICE)
        else:
            self.latent_mu = torch_randn2d(1, self.dim) * self.prior_sigma + self.prior_mu

    def gen_batch_obs(self, num_obs):
        obs = torch_randn2d(num_obs, self.dim) * self.l_sigma + self.latent_mu
        return obs

    def log_prior(self, mu):
        return DiagMvn.log_pdf(mu, self.prior_mu, self.prior_sigma)

    def grad_log_prior(self, mu):
        return DiagMvn.grad_x_log_pdf(mu, self.prior_mu, self.prior_sigma)

    def log_likelihood(self, obs, mu):
        assert mu.shape[0] == 1 or obs.shape[0] == 1  # safe bradcasting
        return DiagMvn.log_pdf(obs, mu, self.l_sigma)

    def grad_log_likelihood(self, obs, mu):
        # will take sum of grad_log_likelihood if multiple obs
        # return tensor of the same shape as mu
        return DiagMvn.grad_mu_log_pdf(obs, mu, self.l_sigma)

    def get_true_posterior(self, obs):
        n = float(obs.shape[0])
        var_prior = self.prior_sigma ** 2
        var_l = self.l_sigma ** 2
        variance = 1.0 / (n / var_l + 1.0 / var_prior)

        x_bar = torch.mean(obs, dim=0, keepdim=True)
        pos_mu = variance * (self.prior_mu / var_prior + n * x_bar / var_l)
        pos_sigma = torch.sqrt(variance)
        return pos_mu, pos_sigma
        
    def log_posterior(self, mu, obs):
        pos_mu, pos_sigma = self.get_true_posterior(obs)
        return DiagMvn.log_pdf(mu, pos_mu, pos_sigma)


class MvnUniModalTestset(MvnUniModalDataset):
    def __init__(self, *args, **kwargs):

        fname = './data/obs_test_dim'+str(cmd_args.gauss_dim)+'_'+str(cmd_args.test_length)+'_%.2f.npy' % cmd_args.l_sigma
        obs = np.load(fname)
        print('%s loaded' % fname)
        self.obs = torch.tensor(obs).to(DEVICE)
        self.epoch = obs.shape[0]
        self.len_seq = obs.shape[1]
        super(MvnUniModalTestset, self).__init__(dim=obs.shape[2], *args, **kwargs)


def onepass_smc(db, ob, particles, alpha):
    num_particles = cmd_args.num_particles

    # re-weight
    reweight = torch.zeros(alpha.shape, dtype=alpha.dtype)
    for ob_i in ob:
        reweight += db.log_likelihood(ob_i.reshape([1, -1]), particles).view(reweight.size())
    reweight = torch.exp(reweight)
    alpha = alpha * reweight

    # re-sample
    ess = torch.sum(alpha) ** 2 / torch.sum(alpha * alpha)

    if ess <= cmd_args.threshold * num_particles:
        alpha = alpha / torch.sum(alpha)
        mnrnd = torch.distributions.multinomial.Multinomial(num_particles, torch.reshape(torch.abs(alpha), [num_particles]))
        idx = mnrnd.sample().int()
        nonzero_cluster = idx.nonzero()
        indx = 0
        new_xi = torch.zeros(particles.shape, dtype=particles.dtype).to(DEVICE)
        for iter in range(nonzero_cluster.shape[0]):
            nonzero_idx = nonzero_cluster[iter][0]
            new_xi[indx: indx + idx[nonzero_idx], :] = particles[nonzero_cluster[iter][0]].repeat(int(idx[nonzero_idx]), 1)
            indx += idx[nonzero_cluster[iter][0]]

        # generate new location
        kde = KDE(new_xi)
        xi = kde.get_samples(cmd_args.num_particles)
        alpha = torch.ones([num_particles, 1], dtype=torch.float) / num_particles
    else:
        alpha = alpha / torch.sum(alpha)
        xi = particles

    return xi, alpha


def sis_r(db, ob, particles):
    num_particles = cmd_args.num_particles
    weights = torch.ones([num_particles, 1], dtype=torch.float).to(DEVICE) / num_particles
    # re-weight
    reweight = torch.zeros(weights.shape, dtype=weights.dtype).to(DEVICE)
    for ob_i in ob:
        reweight += db.log_likelihood(ob_i.reshape([1, -1]), particles)
    reweight = torch.exp(reweight)
    weights = weights * reweight

    # re-sample
    weights = weights / torch.sum(weights)
    mnrnd = torch.distributions.multinomial.Multinomial(num_particles, torch.reshape(torch.relu(weights), [num_particles]))
    idx = mnrnd.sample().int()
    nonzero_cluster = idx.nonzero()
    indx = 0
    new_xi = torch.zeros(particles.shape, dtype=particles.dtype)
    for iter in range(nonzero_cluster.shape[0]):
        nonzero_idx = nonzero_cluster[iter][0]
        new_xi[indx: indx + idx[nonzero_idx], :] = particles[nonzero_cluster[iter][0]].repeat(int(idx[nonzero_idx]), 1)
        indx += idx[nonzero_cluster[iter][0]]
    particles = new_xi

    ess = torch.sum(weights) ** 2 / torch.sum(weights * weights)

    if ess <= cmd_args.threshold * num_particles:
        # generate new location
        kde = KDE(particles)
        particles = kde.get_samples(cmd_args.num_particles)

    return particles


class KernelBR(object):
    def __init__(self,
                 kernel_type='gaussian',
                 dim=cmd_args.gauss_dim,
                 eps=1e-2,
                 delta=1e-2,
                 num_particles=cmd_args.num_particles):

        self.num_particles = num_particles
        self.kernel_type = kernel_type
        self.kernel = self.kernel_function(kernel_type)
        self.dim = dim
        self.eps = eps
        self.delta = delta
        self.data_x = []
        self.data_y = []
        self.gram_x = []
        self.gram_y = []

    @staticmethod
    def kernel_function(kernel_type):
        kernel_dict = {'gaussian': sk_metric.rbf_kernel,
                        'laplacian': sk_metric.laplacian_kernel,
                        'sigmoid': sk_metric.sigmoid_kernel,
                        'polynomial': sk_metric.polynomial_kernel,
                        'cosine': sk_metric.cosine_similarity,
                        'chi2': sk_metric.chi2_kernel
                        }
        return kernel_dict[kernel_type]

    def gram_matrix(self, x, y):
        # compute bandwith
        if self.kernel_type == 'gaussian':
            ind = [random.randrange(0, len(x)) for _ in range(0, 1000)]
            xx = x[ind]
            ind = [random.randrange(0, len(y)) for _ in range(0, 1000)]
            yy = y[ind]
            dist = xx - yy
            dist = torch.sqrt(torch.sum(dist * dist, 1))
            gram_m = self.kernel(x.cpu().numpy(), y.cpu().numpy(), torch.median(dist).cpu().numpy())
        else:
            gram_m = self.kernel(x.cpu().numpy(), y.cpu().numpy())

        return torch.tensor(gram_m).to(DEVICE)

    def reweight(self, gram_x, gram_y, alpha, k_y, n):
        reg_gram = gram_x + n * self.eps * torch.eye(n).to(DEVICE)
        while True:
            try:
                inv_gram = torch.inverse(reg_gram)
            except Exception:
                reg_gram += 10 * n * self.eps * torch.eye(n).to(DEVICE)
            else:
                break
        diag_entry = torch.mv(inv_gram, torch.mv(gram_x, alpha)) * n
        diag_gram_y = diag_entry.view(-1, 1) * gram_y
        reg_gram = torch.mm(diag_gram_y, diag_gram_y) + self.delta * torch.eye(n).to(DEVICE)
        while True:
            try:
                inv_gram = torch.inverse(reg_gram)
            except Exception:
                reg_gram += 10 * self.delta * torch.eye(n).to(DEVICE)
            else:
                break
        weight = diag_entry * k_y
        weight = torch.mv(inv_gram, weight)
        weight = torch.mv(diag_gram_y, weight)
        return weight

    def filter(self, ob, alpha):
        # kernel mean
        n = self.num_particles
        alpha = alpha / torch.sum(alpha)

        k_y = self.gram_matrix(self.data_y, ob).view(-1)
        alpha = self.reweight(self.gram_x, self.gram_y, alpha, k_y, n)
        return alpha

    def resampling(self, weights, particles):
        n = self.num_particles
        x = torch.zeros([n, self.dim], dtype=torch.float).to(DEVICE)

        weights = weights.view(-1)
        gram = self.gram_matrix(particles, particles)
        weight_gram = torch.mv(gram, weights)
        ind = torch.argmax(weight_gram)
        x[0] = particles[ind]

        for p in range(1, n):
            x_t = x[0:p]
            gram = self.gram_matrix(particles, x_t)
            gram = torch.sum(gram, 1)
            ind = torch.argmax(weight_gram - gram)
            x[p] = particles[ind]

        return x

    def training(self, db):
        num_particles = self.num_particles
        xx = torch.zeros([num_particles, cmd_args.gauss_dim], dtype=torch.float).to(DEVICE)
        yy = torch.zeros([num_particles, cmd_args.gauss_dim], dtype=torch.float).to(DEVICE)

        for i in range(num_particles):
            db._reset(mu_given=None)
            xx[i:i+1] = db.latent_mu
            yy[i:i+1] = db.gen_batch_obs(num_obs=1)

        self.data_x = xx
        self.data_y = yy
        self.gram_x = self.gram_matrix(xx, xx)
        self.gram_y = self.gram_matrix(yy, yy)


class KernelBRwR(KernelBR):

    def __init__(self):
        super(KernelBRwR, self).__init__()

    def resampling(self, weights, particles):
        n = self.num_particles
        x = torch.zeros([n, self.dim], dtype=torch.float)

        weights = weights.view(-1)
        gram = self.gram_matrix(particles, particles)
        weight_gram = torch.mv(gram, weights)
        ind = torch.argmax(weight_gram)
        x[0] = particles[ind]

        for p in range(1, n):
            x_t = x[0:p]
            gram = self.gram_matrix(particles, x_t)
            gram = torch.sum(gram, 1)
            ind = torch.argmax(weight_gram - gram)
            x[p] = particles[ind]

        return x

    def filter_w_resample(self, ob, alpha, prior_particles):
        # kernel mean
        n = self.num_particles
        alpha = alpha / torch.sum(alpha)
        k_x_u = self.gram_matrix(self.data_x, prior_particles)
        kernel_m = torch.mv(k_x_u, alpha)
        k_y = self.gram_matrix(self.data_y, ob).view(-1)
        alpha = self.reweight(self.gram_x, self.gram_y, kernel_m, k_y, n)
        return alpha

    def reweight(self, gram_x, gram_y, kernel_m, k_y, n):
        reg_gram = gram_x + n * self.eps * torch.eye(n)
        while True:
            try:
                inv_gram = torch.inverse(reg_gram)
            except Exception:
                reg_gram += 5 * n * self.eps * torch.eye(n)
            else:
                break
        diag_entry = torch.mv(inv_gram, kernel_m) * n
        diag_gram_y = diag_entry.view(-1, 1) * gram_y
        reg_gram = torch.mm(diag_gram_y, diag_gram_y) + self.delta * torch.eye(n)
        while True:
            try:
                inv_gram = torch.inverse(reg_gram)
            except Exception:
                reg_gram += 5 * self.delta * torch.eye(n)
            else:
                break
        weight = diag_entry * k_y
        weight = torch.mv(inv_gram, weight)
        weight = torch.mv(diag_gram_y, weight)
        return weight
