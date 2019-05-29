from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import random
import torch
import pickle

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import chain
from tqdm import tqdm
from pfbayes.common.cmd_args import cmd_args
from pfbayes.common.consts import DEVICE
from pfbayes.common.distributions import KDE, torch_randn2d
from pfbayes.common.distributions import DiagMvn
from pfbayes.common.ode.misc import build_model

from pfbayes.experiments.mvn_unimodal.utils import MvnUniModalDataset
from pfbayes.common.train_utils import train_global_x_loop, forward_global_x, seg_train_global_x_loop
from pfbayes.common.plot_utils.visualize_flow import get_flow_heatmaps, get_true_heatmaps, get_normalized_heatmaps
from pfbayes.common.plot_utils.plt_helper import plot_image_seqs, create_video
from pfbayes.experiments.mvn_unimodal.test_utils import eval_loop


def eval_flow(flow, mvn_dist, val_db):
    flow.eval()
    val_gen = val_db.data_gen(batch_size=1,
                              phase='val',
                              auto_reset=False,
                              shuffle=False)
    ent = 0.0
    for n_s in tqdm(range(cmd_args.num_vals)):
        hist_obs = []
        particles = mvn_dist.get_samples(cmd_args.num_particles)
        densities = mvn_dist.get_log_pdf(particles)        
        for t, ob in enumerate(val_gen):
            particles, densities = flow(particles, densities, 
                                        prior_samples=particles,
                                        ob_m=ob)
            hist_obs.append(ob)
            with torch.no_grad():
                pos_mu, pos_sigma = db.get_true_posterior(torch.cat(hist_obs, dim=0))
                q_mu = torch.mean(particles, dim=0, keepdim=True)
                q_std = torch.std(particles, dim=0, keepdim=True)
                if n_s + 1 == cmd_args.num_vals:
                    print('step:', t)
                    print('true posterior:', pos_mu.cpu().data.numpy(), pos_sigma.cpu().data.numpy())
                    print('estimated:', q_mu.cpu().data.numpy(), q_std.cpu().data.numpy())

                p_particles = torch_randn2d(cmd_args.num_mc_samples, val_db.dim) * pos_sigma + pos_mu
                kde = KDE(particles)
                cur_ent = -torch.mean(kde.log_pdf(p_particles)).item()
                if n_s + 1 == cmd_args.num_vals:
                    print('cross entropy:', cur_ent)
                ent += cur_ent
            if t + 1 == cmd_args.train_samples:
                break
    print('avg ent over %d seqs: %.4f' % (cmd_args.num_vals, ent/cmd_args.num_vals))
    flow.train()
    return ent


def vis_flow(flow, mvn_dist, val_db):
    flow.eval()
    w = 100
    x = np.linspace(-5, 5, w)
    y = np.linspace(-5, 5, w)
    xx, yy = np.meshgrid(x, y)
    mus = np.stack([xx.flatten(), yy.flatten()]).transpose()
    mus = torch.Tensor(mus.astype(np.float32)).to(DEVICE)

    lm_val_gen = lambda : val_db.data_gen(batch_size=1, 
                                          phase='val',
                                          auto_reset=False,
                                          shuffle=False)
    log_scores = val_db.log_prior(mus)
    scores = torch.softmax(log_scores.view(-1), -1).view(w, w).data.cpu().numpy()

    flow_heats = [scores] + get_flow_heatmaps(flow, mvn_dist, lm_val_gen, val_db, mus)
    true_heats = [scores] + get_true_heatmaps(mvn_dist, lm_val_gen, val_db, mus)
    images = list(zip(flow_heats, true_heats))
    save_prefix = os.path.join(cmd_args.save_dir, 'heat-step')
    plot_image_seqs(images, save_prefix)
    create_video(save_prefix,
                 output_name=os.path.join(cmd_args.save_dir, 'traj.mp4'))


def lm_train_gen(db, batch_size):
    db.reset()
    return db.data_gen(batch_size=batch_size,
                       phase='train',
                       auto_reset=False,
                       shuffle=True)


def get_init_dist(db, batch_size, cmd_args, mvn_dist):
    db.reset()
    data_gen = db.data_gen(batch_size=batch_size,
                           phase='train',
                           auto_reset=False,
                           shuffle=True)
    hist_obs = []
    assert cmd_args.stage_len >= 1 and cmd_args.stage_len <= cmd_args.train_samples    
    num_prior_ob = np.random.randint(cmd_args.train_samples - cmd_args.stage_len + 1)
    for i in range(num_prior_ob):
        ob = next(data_gen)
        hist_obs.append(ob)
    if len(hist_obs) == 0:
        dist = mvn_dist
        func_log_prior = lambda new_x, old_x: db.log_prior(new_x)
    else:
        pos_mu, pos_sigma = db.get_true_posterior(torch.cat(hist_obs, dim=0))
        dist = DiagMvn(pos_mu, pos_sigma)
        func_log_prior = lambda new_x, old_x: DiagMvn.log_pdf(new_x, pos_mu, pos_sigma)
    return data_gen, dist, func_log_prior


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    db = MvnUniModalDataset(dim=cmd_args.gauss_dim, 
                            prior_mu=cmd_args.prior_mu,
                            prior_sigma=cmd_args.prior_sigma,
                            l_sigma=cmd_args.l_sigma,
                            partition_sizes={'train': cmd_args.train_samples})
    val_db = MvnUniModalDataset(dim=cmd_args.gauss_dim, 
                                prior_mu=cmd_args.prior_mu,
                                prior_sigma=cmd_args.prior_sigma,
                                l_sigma=cmd_args.l_sigma,
                                partition_sizes={'val': cmd_args.train_samples * cmd_args.num_vals})  # eval on several sequences

    flow = build_model(cmd_args, x_dim=db.dim, ob_dim=db.dim, ll_func=db.log_likelihood)

    if cmd_args.init_model_dump is not None:
        print('loading', cmd_args.init_model_dump)
        flow.load_state_dict(torch.load(cmd_args.init_model_dump))

    if cmd_args.phase == 'eval_metric':
        sampler = lambda db, ob, p, d: flow(p, d, 
                                            prior_samples=p,
                                            ob_m=ob)
        metric = eval_loop(sampler)
        with open('%s/eval_result.pkl' % cmd_args.save_dir, 'wb') as f:
            pickle.dump(metric, f)
        sys.exit()

    optimizer = optim.Adam(flow.parameters(),
                           lr=cmd_args.learning_rate,
                           weight_decay=cmd_args.weight_decay)

    mvn_dist = DiagMvn(mu=[cmd_args.prior_mu] * cmd_args.gauss_dim,
                       sigma=[cmd_args.prior_sigma] * cmd_args.gauss_dim)

    if cmd_args.phase == 'normal_train':
        train_global_x_loop(cmd_args, lambda x: lm_train_gen(db, x), 
                            prior_dist=mvn_dist,
                            flow=flow,
                            optimizer=optimizer,
                            func_ll=db.log_likelihood,
                            func_log_prior=db.log_prior,
                            eval_func=lambda f, p: eval_flow(f, p, val_db))
    elif cmd_args.phase == 'seg_train':
        seg_train_global_x_loop(cmd_args, lambda x: get_init_dist(db, x, cmd_args, mvn_dist), 
                                flow=flow,
                                optimizer=optimizer,
                                func_ll=db.log_likelihood,
                                eval_func=lambda f, p: eval_flow(f, mvn_dist, val_db))
    elif cmd_args.phase == 'test':
        eval_flow(flow, mvn_dist, val_db)
        vis_flow(flow, mvn_dist, val_db)
    else:
        raise NotImplementedError
