from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys
import pickle
import time
import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import chain
from tqdm import tqdm
from pfbayes.common.cmd_args import cmd_args
from pfbayes.common.consts import DEVICE
from pfbayes.common.distributions import DiagMvn
from pfbayes.common.ode.misc import build_model


from pfbayes.experiments.hmm_lds.utils import HmmLdsDataset
from pfbayes.experiments.hmm_lds.test_utils import HmmLdsTestset
from pfbayes.common.distributions import KDE
from pfbayes.common.train_utils import train_local_x_loop, forward_local_x
from pfbayes.common.plot_utils.visualize_flow import get_flow_heatmaps, get_true_heatmaps, get_normalized_heatmaps
from pfbayes.common.plot_utils.plt_helper import plot_image_seqs, create_video
from pfbayes.experiments.hmm_lds.test_utils import eval_loop


def eval_flow(flow, mvn_dist, db, offline_val_list):
    flow.eval()
    ent = 0.0
    for idx, offline_val in enumerate(offline_val_list):
        val_gen = iter(offline_val)
        particles = mvn_dist.get_samples(cmd_args.num_particles)
        densities = mvn_dist.get_log_pdf(particles)    
        
        pos_mu = db.prior_mu
        pos_cov = db.prior_sigma * db.prior_sigma
        pos_cov = torch.diag(pos_cov.reshape(db.dim))
        
        for t, ob in enumerate(val_gen):
            particles, densities = flow(particles, densities, 
                                        prior_samples=particles,
                                        ob_m=ob)
            # evaluate
            pos_mu, pos_cov = db.get_true_new_posterior(ob, pos_mu, pos_cov)
            if idx == 0:
                print('step:', t)
                print('true posterior:', pos_mu.cpu().data.numpy(), pos_cov.cpu().data.numpy())
                print('estimated:', np.mean(particles.cpu().data.numpy(), axis=0), np.cov(particles.cpu().data.numpy().transpose()))

            p_particles = np.random.multivariate_normal(pos_mu.data.cpu().numpy().flatten(), 
                                                        pos_cov.data.cpu().numpy(), 
                                                        cmd_args.num_mc_samples).astype(np.float32)
            kde = KDE(particles)
            cur_ent = -torch.mean(kde.log_pdf(torch.tensor(p_particles).to(DEVICE))).item()
            if idx == 0:
                print('cross entropy:', cur_ent)
            ent += cur_ent
    ent /= len(offline_val_list)
    print('avg ent over %d seqs: %.4f' % (len(offline_val_list), ent))
    flow.train()
    return ent


def vis_flow(flow, mvn_dist, db, offline_val):
    flow.eval()
    w = 100
    x = np.linspace(-3, 3, w)
    y = np.linspace(-3, 3, w)
    xx, yy = np.meshgrid(x, y)
    mus = np.stack([xx.flatten(), yy.flatten()]).transpose()
    mus = torch.Tensor(mus.astype(np.float32)).to(DEVICE)
    
    log_scores = mvn_dist.get_log_pdf(mus)
    scores = torch.softmax(log_scores.view(-1), -1).view(w, w).data.cpu().numpy()
    
    val_gen = iter(offline_val)

    pos_mu = db.prior_mu
    pos_cov = db.prior_sigma * db.prior_sigma
    pos_cov = torch.diag(pos_cov.reshape(db.dim))
    true_heats = [scores]
    
    particles = mvn_dist.get_samples(cmd_args.num_particles)
    densities = mvn_dist.get_log_pdf(particles)
    kde = KDE(particles)
    log_scores = kde.log_pdf(mus)
    est_scores = torch.softmax(log_scores.view(-1), -1).view(w, w).data.cpu().numpy()
    flow_heats = [est_scores]

    for t, ob in enumerate(val_gen):
        # evaluate
        pos_mu, pos_cov = db.get_true_new_posterior(ob, pos_mu, pos_cov)
        particles, densities = flow(particles, densities, 
                                    prior_samples=particles,
                                    ob_m=ob)
        kde = KDE(particles)
        log_scores = kde.log_pdf(mus)
        est_scores = torch.softmax(log_scores.view(-1), -1).view(w, w).data.cpu().numpy()
        flow_heats.append(est_scores)
        dist = torch.distributions.MultivariateNormal(pos_mu, pos_cov)
        log_scores = dist.log_prob(mus)
        exact_scores = torch.softmax(log_scores.view(-1), -1).view(w, w).data.cpu().numpy()
        true_heats.append(exact_scores)
    images = list(zip(flow_heats, true_heats))
    save_prefix = os.path.join(cmd_args.save_dir, 'heat-step')
    plot_image_seqs(images, save_prefix)
    create_video(save_prefix,
                 output_name=os.path.join(cmd_args.save_dir, 'traj.mp4'))
    flow.train()


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    db = HmmLdsDataset(dim=cmd_args.gauss_dim,
                        prior_mu=cmd_args.prior_mu,
                        prior_sigma=cmd_args.prior_sigma,
                        l_sigma=cmd_args.l_sigma,
                        t_sigma=cmd_args.t_sigma,
                        partition_sizes={'train': cmd_args.train_samples})
    offline_val_list = []
    for i in range(cmd_args.num_vals):
        obs_gen = db.gen_seq_obs(cmd_args.train_samples)
        offline_val = [ob for ob in obs_gen]
        offline_val_list.append(offline_val)

    flow = build_model(cmd_args, x_dim=db.dim, ob_dim=db.dim, ll_func=db.log_likelihood)

    if cmd_args.init_model_dump is not None:
        print('loading', cmd_args.init_model_dump)
        flow.load_state_dict(torch.load(cmd_args.init_model_dump))

    mvn_dist = DiagMvn(mu=[cmd_args.prior_mu] * cmd_args.gauss_dim,
                       sigma=[cmd_args.prior_sigma] * cmd_args.gauss_dim)

    if cmd_args.phase == 'visualize':
        vis_flow(flow, mvn_dist, db, offline_val)
    elif cmd_args.phase == 'eval_metric':
        test_db = HmmLdsTestset()
        tot_time = 0.0
        num_eval = []
        with torch.no_grad():
            for e in tqdm(range(test_db.epoch)):
                ob_seq = [torch.tensor(test_db.obs[e][t]).view(1, -1).to(DEVICE) for t in range(test_db.len_seq)]
                
                particles = mvn_dist.get_samples(cmd_args.num_particles)
                densities = mvn_dist.get_log_pdf(particles)
                t = time.time()
                for ob in ob_seq:
                    particles, densities = flow(particles, densities, prior_samples=particles, ob_m=ob)
                    num_eval.append(flow.chain[0].odefunc._num_evals.item())
                tot_time += time.time() - t
        tot_time /= test_db.epoch
        print('avg eval', np.mean(num_eval))
        print('aveage time %.4f' % tot_time)
        sampler = lambda db, ob, p, d, e, f: flow(p, d, 
                                                  prior_samples=p,
                                                  ob_m=ob)
        metric = eval_loop(sampler)
        with open('%s/eval_result.pkl' % cmd_args.save_dir, 'wb') as f:
            pickle.dump(metric, f)
        sys.exit()
    elif cmd_args.phase == 'train':
        optimizer = optim.Adam(flow.parameters(),
                            lr=cmd_args.learning_rate,
                            weight_decay=cmd_args.weight_decay)

        train_local_x_loop(cmd_args, lambda bsize: db.gen_seq_obs(cmd_args.train_samples),
                           prior_dist=mvn_dist,
                           flow=flow,
                           optimizer=optimizer,
                           func_ll=db.log_likelihood,
                           func_log_transit=db.transition_log_pdf,
                           eval_func=lambda f, p: eval_flow(f, p, db, offline_val_list))
    elif cmd_args.phase == 'val':
        eval_flow(flow, mvn_dist, db, offline_val_list)
    else:
        raise NotImplementedError
