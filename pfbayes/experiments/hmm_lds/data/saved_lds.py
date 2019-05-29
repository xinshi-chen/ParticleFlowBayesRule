import random
import numpy as np
import torch
from pfbayes.common.cmd_args import cmd_args
from pfbayes.common.distributions import torch_randn2d

import pickle


random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)

transition_a = torch_randn2d(cmd_args.gauss_dim, cmd_args.gauss_dim)
likelihood_b = torch_randn2d(cmd_args.gauss_dim, cmd_args.gauss_dim)

filename = 'lds_model10.pkl'
with open(filename, 'wb') as f:
    pickle.dump((transition_a, likelihood_b), f)
