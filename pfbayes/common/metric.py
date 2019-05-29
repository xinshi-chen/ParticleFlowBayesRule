from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import sklearn.metrics.pairwise as sk_metric
from pfbayes.common.distributions import KDE
import torch
from pfbayes.common.cmd_args import cmd_args
from numpy import linalg as LA
import pickle
import os

def square_mmd_fine(p_samples, q_samples, n_p, n_q, kernel_type):
    """
    n_p: number of samples from true distribution p

    assume n_p >> n_q
    """
    kernel_dict = {
        'gaussian': sk_metric.rbf_kernel,
        'laplacian': sk_metric.laplacian_kernel,
        'sigmoid': sk_metric.sigmoid_kernel,
        'polynomial': sk_metric.polynomial_kernel,
        'cosine': sk_metric.cosine_similarity,
    }

    kernel = kernel_dict[kernel_type]

    p_samples = np.array(p_samples)
    q_samples = np.array(q_samples)

    k_xi_xj = kernel(p_samples, p_samples)
    k_yi_yj = kernel(q_samples, q_samples)
    k_xi_yj = kernel(p_samples, q_samples)

    off_diag_k_xi_xj = (np.sum(k_xi_xj) - np.sum(np.diag(k_xi_xj))) / n_p / (n_p - 1)
    off_diag_k_yi_yj = (np.sum(k_yi_yj) - np.sum(np.diag(k_yi_yj))) / n_q / (n_q - 1)
    sum_k_xi_yj = np.sum(k_xi_yj) * 2 / n_p / n_q

    return off_diag_k_xi_xj + off_diag_k_yi_yj - sum_k_xi_yj


def e_p_log_q(p_samples, q_samples):

    q_samples = torch.Tensor(q_samples)
    p_samples = torch.Tensor(p_samples)
    kde = KDE(q_samples)
    log_score = kde.log_pdf(p_samples)
    return torch.mean(log_score)


class EvalMetric(object):
    """
    using numpy
    """
    def __init__(self, particles, true_mean, true_cov, dim, num_true_samples=None):
        self.dim = dim
        self.particles = np.array(particles).reshape(-1, dim)
        self.n_particles = self.particles.shape[0]
        self.true_mean = np.array(true_mean).reshape(dim)
        self.true_cov = np.array(true_cov).reshape(dim, dim)
        if num_true_samples is None:
            self.n_samples = max(5000, 10 * cmd_args.num_particles)
        else:
            self.n_samples = num_true_samples

    def square_mmd(self, kernel_type='gaussian'):
        p_particles = np.random.multivariate_normal(self.true_mean.astype(np.float64), self.true_cov.astype(np.float64), self.n_samples)
        return square_mmd_fine(p_particles, self.particles, self.n_samples, self.n_particles, kernel_type)

    def cross_entropy(self):
        p_particles = np.random.multivariate_normal(self.true_mean.astype(np.float64), self.true_cov.astype(np.float64), self.n_samples)
        return -np.array(e_p_log_q(p_particles, self.particles).cpu())

    def integral_eval(self, test_function):
        full_path = os.path.realpath(__file__)
        path = os.path.dirname(full_path)
        filename = path+'/test_function/test_function_'+str(cmd_args.gauss_dim)+'.pkl'
        with open(filename, 'rb') as f:
            matrix_aa, matrix_a, matrix_b, a, b = pickle.load(f)
        if test_function == 'x':
            return self.dist_of_mean()
        elif test_function == 'xAx':
            return self.distance_of_xax(matrix_aa)
        elif test_function == 'quadratic':
            return self.distance_of_quadratic(matrix_a, a, matrix_b, b)
        else:
            print('test function not supported')

    def dist_of_mean(self, q_samples=None):
        if q_samples is None:
            q_samples = self.particles
        else:
            q_samples = np.array(q_samples).reshape(-1, self.dim)
        q_mean = np.mean(q_samples, 0)
        return LA.norm(q_mean-self.true_mean)

    def distance_of_xax(self, matrix_a, q_samples=None):
        """||E_q[x'Ax] - E_p[x'Ax]||"""
        if q_samples is None:
            q_samples = self.particles
        else:
            q_samples = np.array(q_samples).reshape(-1, self.dim)
        mean = self.true_mean.reshape(1, self.dim)
        true_val = np.trace(np.matmul(matrix_a, self.true_cov))
        true_val += np.sum(np.dot(mean, np.matmul(matrix_a, mean.T)))

        est_ax = np.matmul(matrix_a, q_samples.T)
        est_xax = np.diag(np.matmul(q_samples, est_ax))
        est_xax = np.mean(est_xax)

        return np.abs(true_val - est_xax)

    def distance_of_quadratic(self, matrix_a, a, matrix_b, b, q_samples=None):
        """||E_q[(Ax+a)'(Bx+b)] - E_p[~~~]||"""
        if q_samples is None:
            q_samples = self.particles
        else:
            q_samples = np.array(q_samples).reshape(-1, self.dim)
        # format
        matrix_a = np.array(matrix_a)
        matrix_b = np.array(matrix_b)

        true_val = np.trace(np.matmul(np.matmul(matrix_a, self.true_cov), matrix_b.T))
        true_val += np.dot(matrix_a.dot(self.true_mean) + a, matrix_b.dot(self.true_mean) + b)

        est_ax_a = np.matmul(q_samples, matrix_a.T) + a
        est_bx_b = np.matmul(q_samples, matrix_b.T) + b
        est_val = np.diag(np.matmul(est_ax_a, est_bx_b.T))
        est_val = np.mean(est_val)

        return np.abs(true_val - est_val)


def create_metric_dict(num_epoch, len_sequence):
    metric = dict()
    metric['mmd'] = dict()
    metric['mmd']['gaussian'] = np.zeros([num_epoch, len_sequence], dtype=np.float32)
    metric['mmd']['laplacian'] = np.zeros([num_epoch, len_sequence], dtype=np.float32)
    metric['mmd']['sigmoid'] = np.zeros([num_epoch, len_sequence], dtype=np.float32)
    metric['mmd']['polynomial'] = np.zeros([num_epoch, len_sequence], dtype=np.float32)
    metric['mmd']['cosine'] = np.zeros([num_epoch, len_sequence], dtype=np.float32)
    metric['cross-entropy'] = np.zeros([num_epoch, len_sequence], dtype=np.float32)
    metric['integral-eval'] = dict()
    metric['integral-eval']['x'] = np.zeros([num_epoch, len_sequence], dtype=np.float32)
    metric['integral-eval']['xAx'] = np.zeros([num_epoch, len_sequence], dtype=np.float32)
    metric['integral-eval']['quadratic'] = np.zeros([num_epoch, len_sequence], dtype=np.float32)
    print('evaluate MMD, cross-entropy and discrepancy of integral evaluations')
    return metric


class EvalMetricKbr(object):
    """
    using numpy
    """
    def __init__(self, weights, particles, equal_particles, true_mean, true_cov, dim, num_true_samples=None):
        self.dim = dim
        self.particles = np.array(particles).reshape(-1, dim)
        self.n_particles = self.particles.shape[0]
        self.true_mean = np.array(true_mean).reshape(dim)
        self.true_cov = np.array(true_cov).reshape(dim, dim)
        self.weights = np.array(weights).reshape(-1)
        self.equal_particles = np.array(equal_particles).reshape(-1, dim)
        if num_true_samples is None:
            self.n_samples = max(5000, 10 * cmd_args.num_particles)
        else:
            self.n_samples = num_true_samples

    def square_mmd(self, kernel_type='gaussian'):
        p_particles = np.random.multivariate_normal(self.true_mean.astype(np.float64), self.true_cov.astype(np.float64), self.n_samples)
        return square_mmd_fine(p_particles, self.equal_particles, self.n_samples, self.n_particles, kernel_type)

    def cross_entropy(self):
        p_particles = np.random.multivariate_normal(self.true_mean.astype(np.float64), self.true_cov.astype(np.float64), self.n_samples)
        return -np.array(e_p_log_q(p_particles, self.equal_particles).cpu())

    def integral_eval(self, test_function):
        full_path = os.path.realpath(__file__)
        path = os.path.dirname(full_path)
        filename = path+'/test_function/test_function.pkl'
        with open(filename, 'rb') as f:
            matrix_aa, matrix_a, matrix_b, a, b = pickle.load(f)
        if test_function == 'x':
            return self.dist_of_mean()
        elif test_function == 'xAx':
            return self.distance_of_xax(matrix_aa)
        elif test_function == 'quadratic':
            return self.distance_of_quadratic(matrix_a, a, matrix_b, b)
        else:
            print('test function not supported')

    def dist_of_mean(self):
        q_mean = np.sum(self.weights.reshape(-1, 1) * self.particles, 0)

        return LA.norm(q_mean-self.true_mean)

    def distance_of_xax(self, matrix_a):
        """||E_q[x'Ax] - E_p[x'Ax]||"""

        q_samples = self.particles
        mean = self.true_mean.reshape(1, self.dim)
        true_val = np.trace(np.matmul(matrix_a, self.true_cov))
        true_val += np.sum(np.dot(mean, np.matmul(matrix_a, mean.T)))

        est_ax = np.matmul(matrix_a, q_samples.T)
        est_xax = np.diag(np.matmul(q_samples, est_ax))
        est_xax = np.sum(self.weights.reshape(-1) * est_xax)

        return np.abs(true_val - est_xax)

    def distance_of_quadratic(self, matrix_a, a, matrix_b, b):
        """||E_q[(Ax+a)'(Bx+b)] - E_p[~~~]||"""

        q_samples = self.particles
        # format
        matrix_a = np.array(matrix_a)
        matrix_b = np.array(matrix_b)

        true_val = np.trace(np.matmul(np.matmul(matrix_a, self.true_cov), matrix_b.T))
        true_val += np.dot(matrix_a.dot(self.true_mean) + a, matrix_b.dot(self.true_mean) + b)

        est_ax_a = np.matmul(q_samples, matrix_a.T) + a
        est_bx_b = np.matmul(q_samples, matrix_b.T) + b
        est_val = np.diag(np.matmul(est_ax_a, est_bx_b.T))

        est_val = np.sum(self.weights.reshape(-1) * est_val)

        return np.abs(true_val - est_val)
