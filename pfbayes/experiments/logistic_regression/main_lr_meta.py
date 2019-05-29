from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from itertools import chain
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from pfbayes.common.cmd_args import cmd_args
from pfbayes.common.distributions import KDE, MMD
from pfbayes.common.train_utils import supervised_train_loop as train_loop
from pfbayes.experiments.logistic_regression.lr_utils import MnistDataset, eval_particles_acc, RotateMnistDataset
from pfbayes.common.ode.misc import build_model
from pfbayes.common.ode.ode_nets import KernelEmbedNet, SizeBalancedEmbedNet
from pfbayes.common.distributions import DiagMvn
from pfbayes.common.consts import DEVICE


def eval_func(num_obs, particles, db, phase, prior_dist, flow, ob_net, f_log):
    val_gen = db.data_gen(batch_size=cmd_args.batch_size,
                          phase=phase,
                          auto_reset=False,
                          shuffle=True)
    particles = prior_dist.get_samples(cmd_args.num_particles)
    densities = prior_dist.get_log_pdf(particles)
    acc_list = []
    for step, ob in enumerate(val_gen):        
        feats, labels = ob        
        ob = ob_net(ob)
        pred = torch.sigmoid(torch.matmul(feats, particles.t()))
        pred = torch.mean(pred, dim=-1, keepdim=True)
        acc = (labels < 0.5) == (pred < 0.5)
        acc = torch.mean(acc.float())
        acc_list.append(acc.item())
        msg = 'accuracy at step %d: %.6f' % (step, acc.item())
        if phase == 'test':
            print(msg)
            f_log.write(msg + '\n')        
        new_particles, new_densities = flow(particles, densities,
                                            prior_samples=particles,
                                            ob_m=ob)
        particles = new_particles.detach()
        densities = new_densities.detach()
    return 1.0 - np.mean(acc_list)


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    if cmd_args.meta_type == 'single':
        db = MnistDataset(cmd_args.data_folder)
        ob_net = KernelEmbedNet(db.ob_dim, str(db.ob_dim), cmd_args.nonlinearity, trainable=True).to(DEVICE)
    elif cmd_args.meta_type == 'angle':
        db = RotateMnistDataset(cmd_args.data_folder,
                                max_angle=cmd_args.max_angle,
                                num_vals=cmd_args.num_vals,
                                dim_x=cmd_args.dim_x,
                                dim_y=cmd_args.dim_y)
        ob_net = SizeBalancedEmbedNet(50, 1, '51-51', cmd_args.nonlinearity, trainable=True).to(DEVICE)
    else:
        raise NotImplementedError

    flow = build_model(cmd_args, x_dim=db.x_dim, ob_dim=db.ob_dim, ll_func=db.log_likelihood)
    
    if cmd_args.init_model_dump is not None:
        print('loading', cmd_args.init_model_dump)
        flow.load_state_dict(torch.load(cmd_args.init_model_dump))

    optimizer = optim.Adam(chain(flow.parameters(), ob_net.parameters()),
                           lr=cmd_args.learning_rate,
                           weight_decay=cmd_args.weight_decay)

    prior_dist = DiagMvn(mu=[cmd_args.prior_mu] * db.x_dim,
                         sigma=[cmd_args.prior_sigma] * db.x_dim)

    #  suppose now we only consider stage-wise training, but not minibatch
    if cmd_args.phase == 'train':
        val_len = db.data_partitions['val-0'][0].shape[0]
        cmd_args.stage_len = int(np.ceil(val_len / cmd_args.n_stages / cmd_args.batch_size))
        print('%d stages, each stage has %d steps with %d obs per step' % (cmd_args.n_stages,
                                                                           cmd_args.stage_len,
                                                                           cmd_args.batch_size))
        f_log = open('%s/log-seed-%d.txt' % (cmd_args.save_dir, cmd_args.seed), 'w', buffering=1)
        train_loop(cmd_args, db, prior_dist, flow, ob_net, 
                   coeff=1.0,
                   eval_func=lambda a, b, c, d: eval_func(a, b, c, d, prior_dist, flow, ob_net, f_log))
    elif cmd_args.phase == 'test':
        f_log = open('%s/log-seed-%d.txt' % (cmd_args.save_dir, cmd_args.seed), 'w', buffering=1)
        eval_func(None, None, db, 'test', prior_dist, flow, ob_net, f_log)
        f_log.close()