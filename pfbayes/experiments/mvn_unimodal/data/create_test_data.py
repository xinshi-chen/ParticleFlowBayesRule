
import random
import numpy as np
from pfbayes.common.cmd_args import cmd_args
from pfbayes.experiments.mvn_unimodal.utils import MvnUniModalDataset
import torch


def create_test_data(filename, db, epoch,
                     length_seq=cmd_args.test_length,
                     ):

    obs = np.zeros([epoch, length_seq, cmd_args.gauss_dim], dtype=np.float32)

    for e in range(epoch):
        db.reset()
        obs[e] = db.gen_batch_obs(num_obs=length_seq)

    # save
    try:
        _ = np.load(filename)
    except (OSError, IOError):
        pass

    np.save(filename, obs)


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    torch.set_grad_enabled(False)

    db = MvnUniModalDataset(dim=cmd_args.gauss_dim,
                            prior_mu=cmd_args.prior_mu,
                            prior_sigma=cmd_args.prior_sigma,
                            l_sigma=cmd_args.l_sigma)
    file = './obs_test_dim%d_%d_%.2f.npy' % (cmd_args.gauss_dim, cmd_args.test_length, cmd_args.l_sigma)
    create_test_data(file, db, epoch=cmd_args.test_epoch)
