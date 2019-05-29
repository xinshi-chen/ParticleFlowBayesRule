from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import random
import torch
import pickle
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from pfbayes.common.cmd_args import cmd_args
from pfbayes.common.consts import DEVICE
from pfbayes.common.distributions import DiagMvn, KDE
from pfbayes.common.ode.misc import build_model

from pfbayes.experiments.two_gaussian.utils import TwoGaussDataset
from pfbayes.common.train_utils import train_global_x_loop, forward_global_x
from pfbayes.common.plot_utils.visualize_flow import get_flow_heatmaps, get_normalized_heatmaps
from pfbayes.common.plot_utils.plt_helper import plot_image_seqs, create_video


def eval_flow(cmd_args, flow, mvn_dist, val_db):
    flow.eval()
    val_set = val_db.data_gen(batch_size=cmd_args.batch_size,
                              phase='val',
                              auto_reset=False,
                              shuffle=False)
    loss = 0.0
    for n_s in tqdm(range(cmd_args.num_vals)):
        ob_list = []
        for _ in range(cmd_args.train_samples):
            ob_list.append(next(val_set))
        val_gen = iter(ob_list)
        particles = mvn_dist.get_samples(cmd_args.num_particles)
        densities = mvn_dist.get_log_pdf(particles)

        loss += forward_global_x(flow, particles, densities, val_gen,
                                 val_db.log_likelihood,
                                 val_db.log_prior)[0].item()
    loss /= cmd_args.num_vals
    print('avg loss of %d seqs: %.4f' % (cmd_args.num_vals, loss))
    flow.train()
    return loss


def vis_flow(flow, mvn_dist, val_db):
    flow.eval()
    w = 100
    x = np.linspace(-3, 3, w)
    y = np.linspace(-3, 3, w)
    xx, yy = np.meshgrid(x, y)
    mus = np.stack([xx.flatten(), yy.flatten()]).transpose()
    mus = torch.Tensor(mus.astype(np.float32)).to(DEVICE)

    val_set = val_db.data_gen(batch_size=cmd_args.batch_size,
                              phase='val',
                              auto_reset=False,
                              shuffle=False)
    
    ob_list = []
    for _ in range(cmd_args.train_samples):
        ob_list.append(next(val_set))
    lm_val_gen = lambda: iter(ob_list)

    particles = mvn_dist.get_samples(cmd_args.num_particles)
    densities = mvn_dist.get_log_pdf(particles)
    kde = KDE(particles)
    log_scores = kde.log_pdf(mus)
    est_scores = torch.softmax(log_scores.view(-1), -1).view(w, w).data.cpu().numpy()
    flow_heats = [est_scores]
    val_gen = lm_val_gen()
    for t, ob in enumerate(val_gen):
        # evaluate
        particles, densities = flow(particles, densities, 
                                    prior_samples=particles,
                                    ob_m=ob)
        kde = KDE(particles)
        log_scores = kde.log_pdf(mus)
        est_scores = torch.softmax(log_scores.view(-1), -1).view(w, w).data.cpu().numpy()
        flow_heats.append(est_scores)

    log_scores = val_db.log_prior(mus)
    scores = torch.softmax(log_scores.view(-1), -1).view(w, w).data.cpu().numpy()
    true_heats = [scores] + get_normalized_heatmaps(mvn_dist, lm_val_gen, val_db, mus)

    out_dir = os.path.join(cmd_args.save_dir, 'video-%d' % cmd_args.seed)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    np.save(os.path.join(out_dir, 'flow_heats.npy'), flow_heats)
    np.save(os.path.join(out_dir, 'true_heats.npy'), true_heats)

    images = list(zip(flow_heats, true_heats))
    save_prefix = os.path.join(out_dir, 'heat-step')
    plot_image_seqs(images, save_prefix)
    create_video(save_prefix,
                 output_name=os.path.join(out_dir, 'traj.mp4'))


def lm_train_gen(db, batch_size):
    db.reset(mu_given=[-1, 2])
    return db.data_gen(batch_size=batch_size,
                       phase='train',
                       auto_reset=False,
                       shuffle=True)


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    db = TwoGaussDataset(prior_mu=cmd_args.prior_mu,
                         prior_sigma=cmd_args.prior_sigma, 
                         mu_given=[-1, 2],
                         l_sigma=1.0,
                         p=0.5,
                         partition_sizes={'train': cmd_args.train_samples})
    val_db = TwoGaussDataset(prior_mu=cmd_args.prior_mu,  
                             prior_sigma=cmd_args.prior_sigma, 
                             mu_given=[-1, 2],
                             l_sigma=1.0,
                             p=0.5,
                             partition_sizes={'val': cmd_args.train_samples * cmd_args.num_vals})

    flow = build_model(cmd_args, x_dim=2, ob_dim=db.dim)

    if cmd_args.init_model_dump is not None:
        print('loading', cmd_args.init_model_dump)
        flow.load_state_dict(torch.load(cmd_args.init_model_dump))

    mvn_dist = DiagMvn(mu=[cmd_args.prior_mu] * cmd_args.gauss_dim,
                       sigma=[cmd_args.prior_sigma] * cmd_args.gauss_dim)

    if cmd_args.phase == 'train':
        optimizer = optim.Adam(flow.parameters(),
                            lr=cmd_args.learning_rate,
                            weight_decay=cmd_args.weight_decay)
        train_global_x_loop(cmd_args, lambda x: lm_train_gen(db, x), 
                prior_dist=mvn_dist,
                flow=flow,
                optimizer=optimizer,
                func_ll=db.log_likelihood,
                func_log_prior=db.log_prior,
                eval_func=lambda f, p: eval_flow(cmd_args, f, p, val_db))
    eval_flow(cmd_args, flow, mvn_dist, val_db)
    vis_flow(flow, mvn_dist, val_db)
