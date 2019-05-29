
import random
import numpy as np
from pfbayes.common.cmd_args import cmd_args
from pfbayes.experiments.hmm_lds.utils import HmmLdsDataset
from pfbayes.common.distributions import DiagMvn
import torch


def create_test_data(filename, db, epoch,
                     length_seq=cmd_args.test_length,
                     ):

    obs = np.zeros([epoch, length_seq, cmd_args.gauss_dim], dtype=np.float32)

    for e in range(epoch):
        obs_gen = db.gen_seq_obs(length_seq)
        for t, ob in enumerate(obs_gen):
            obs[e][t] = np.array(ob)

    # save
    try:
        _ = np.load(filename)
    except (OSError, IOError):
        pass
    else:
        print('ATTENTION: there exists saved test data, but now it is overwritten.')

    np.save(filename, obs)


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    torch.set_grad_enabled(False)

    dbhmm = HmmLdsDataset(dim=cmd_args.gauss_dim,
                       prior_mu=cmd_args.prior_mu,
                       prior_sigma=cmd_args.prior_sigma,
                       l_sigma=cmd_args.l_sigma,
                       t_sigma=cmd_args.t_sigma)
    print('ATTENTION: random seed must be consistent with training phase. otherwise the LDS model will be different.')

    file = './obs_test_dim10_len25.npy'
    create_test_data(file, dbhmm, epoch=cmd_args.test_epoch)
