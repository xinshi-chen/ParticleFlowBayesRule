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
from pfbayes.common.train_utils import supervised_train_loop as train_loop
from pfbayes.experiments.logistic_regression.lr_utils import MnistDataset, eval_particles_acc
from pfbayes.common.ode.misc import build_model
from pfbayes.common.ode.ode_nets import KernelEmbedNet
from pfbayes.common.distributions import DiagMvn
from pfbayes.common.consts import DEVICE


def eval_func(num_obs, particles, db, phase, f_log):
    val_gen = db.data_gen(batch_size=cmd_args.batch_size,
                          phase=phase,
                          auto_reset=False,
                          shuffle=False)
    acc = eval_particles_acc(particles, val_gen)
    print('%s accuracy: %.4f' % (phase, acc))
    if phase == 'test':
        factor = cmd_args.n_stages * cmd_args.batch_size * cmd_args.stage_len
        f_log.write('num_obs %d, loss %.8f\n' % (num_obs, 1.0 - acc))
    return 1.0 - acc


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    db = MnistDataset(cmd_args.data_folder)

    flow = build_model(cmd_args, x_dim=db.x_dim, ob_dim=db.ob_dim, ll_func=db.log_likelihood)
    ob_net = KernelEmbedNet(db.ob_dim, str(db.ob_dim), cmd_args.nonlinearity, trainable=True).to(DEVICE)
    
    if cmd_args.init_model_dump is not None:
        state = torch.load(cmd_args.init_model_dump)
        flow.load_state_dict(state)

    prior_dist = DiagMvn(mu=[cmd_args.prior_mu] * db.x_dim,
                         sigma=[cmd_args.prior_sigma] * db.x_dim)

    test_locs = [100, 200, 300, 400, 600, 700, 800, 1000, 1300,
                 1600, 2000, 2600, 3200, 4000, 5100, 6400, 8000, 10000, 12600, 
                 15900, 20000, 25200, 31700, 39900, 50200, 63100, 79500, 100000,
                 125900, 158500, 199600, 251200, 316300, 398200, 501200, 631000,
                 794400, 1000000, 1259000, 1584900, 1995300, 2511900, 
                 3162300, 3981100, 5011900, 6309600, 7943300]

    if cmd_args.phase == 'train':
        if cmd_args.stage_len <= 0:
            cmd_args.stage_len = 1
        # we need to approximate the posterior of entire dataset using n_stages flows
        # so it is equivalent to have this true_bsize as the actual batch size
        true_bsize = db.num_train / cmd_args.n_stages / cmd_args.stage_len
        # coefficient in front of entropy term
        coeff = cmd_args.batch_size / true_bsize

        f_log = open('%s/log.txt' % cmd_args.save_dir, 'w', buffering=1)
        train_loop(cmd_args, db, prior_dist, flow, ob_net, coeff,
                   eval_func=lambda a, b, c, d: eval_func(a, b, c, d, f_log),
                   test_locs=test_locs)
        f_log.close()
    else:
        raise NotImplementedError
