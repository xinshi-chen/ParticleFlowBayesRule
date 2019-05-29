
import random
import numpy as np
from pfbayes.common.cmd_args import cmd_args
from pfbayes.common.consts import DEVICE

from pfbayes.experiments.hmm_lds.utils import HmmLdsDataset
from pfbayes.common.distributions import DiagMvn
import torch
from pfbayes.experiments.hmm_lds.utils import HmmLdsTestset
from pfbayes.common.metric import create_metric_dict
from pfbayes.common.metric import EvalMetric


def test_prepare():
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    torch.set_grad_enabled(False)
    # load test data
    test_db = HmmLdsTestset()

    # config
    num_particles = cmd_args.num_particles
    len_sequence = test_db.len_seq
    num_epoch = test_db.epoch

    # initial particles from prior
    mvn_dist = DiagMvn(mu=[cmd_args.prior_mu] * cmd_args.gauss_dim,
                           sigma=[cmd_args.prior_sigma] * cmd_args.gauss_dim)

    pos_mu = test_db.prior_mu
    pos_cov = test_db.prior_sigma * test_db.prior_sigma
    pos_cov = torch.diag(pos_cov.reshape(test_db.dim))

    metric = create_metric_dict(num_epoch, len_sequence)

    return test_db, metric, mvn_dist, pos_mu, pos_cov, num_epoch, len_sequence, num_particles


def eval_step(particles, pos_mu, pos_cov, test_db, metric, e, t):
    eval_metric = EvalMetric(
                                particles=particles,
                                true_mean=np.array(pos_mu),
                                true_cov=np.array(pos_cov),
                                dim=test_db.dim,
                                num_true_samples=cmd_args.num_mc_samples,
                                )
    for key in metric['mmd']:
        metric['mmd'][key][e][t] = eval_metric.square_mmd(kernel_type=key)
    for key in metric['integral-eval']:
        metric['integral-eval'][key][e][t] = eval_metric.integral_eval(test_function=key)
    metric['cross-entropy'][e][t] = eval_metric.cross_entropy()
    # print
    print('step:', t, 'epoch', e)
    return metric


def eval_loop(sampler):
    # prepare for test
    test_db, metric, prior_dist, pos_mu, pos_cov, num_epoch, len_sequence, num_particles = test_prepare()

    for e in range(num_epoch):
        particles = prior_dist.get_samples(num_particles)
        densities = prior_dist.get_log_pdf(particles)
        for t in range(len_sequence):
            ob = test_db.obs[e][t]
            ob = torch.tensor(ob).view(1, -1).to(DEVICE)

            # update particles
            particles, densities = sampler(test_db, ob, particles, densities, e, t)

            # update true posterior
            pos_mu, pos_cov = test_db.get_true_new_posterior(ob, pos_mu, pos_cov)
            # evaluate
            metric = eval_step(particles, pos_mu.cpu(), pos_cov.cpu(), test_db, metric, e, t)

        # print
        print('epoch', e+1)
        for key in metric['mmd']:
            print('mmd-'+key+': ', np.mean(metric['mmd'][key][0:e+1], 0))
        for key in metric['integral-eval']:
            print('integral-'+key+': ', np.mean(metric['integral-eval'][key][0:e+1], 0))
        print('cross-entropy: ', np.mean(metric['cross-entropy'][0:e+1], 0))

    # print
    print('final result')

    for key in metric['mmd']:
        print('mmd-'+key+': ', np.mean(metric['mmd'][key], 0))
    for key in metric['integral-eval']:
        print('integral-'+key+': ', np.mean(metric['integral-eval'][key], 0))
    print('cross-entropy: ', np.mean(metric['cross-entropy'], 0))
    return metric
