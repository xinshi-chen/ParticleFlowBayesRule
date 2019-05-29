from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
from argparse import Namespace
import os
import torch
import numpy as np
import logging

from pfbayes.common.consts import NONLINEARITIES, SOLVERS

cmd_opt = argparse.ArgumentParser(description='Argparser for sequential particle flow', allow_abbrev=False)
cmd_opt.add_argument('-gpu', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')
cmd_opt.add_argument('-ode', type=str, default='adjoint', help='ode backprop type', choices=['adjoint', 'naive'])
cmd_opt.add_argument('-save_dir', type=str, default='./scratch', help='save folder')
cmd_opt.add_argument('-phase', type=str, default='train', help='phase of execution', choices=['train', 'seg_train', 'normal_train', 'visualize', 'val', 'test', 'eval_metric'])
cmd_opt.add_argument('-init_model_dump', type=str, default=None, help='initial model dump')
cmd_opt.add_argument('-dropbox', type=str, default=None, help='dropbox folder')
cmd_opt.add_argument('-seed', type=int, default=19260817, help='seed')
cmd_opt.add_argument('-num_epochs', type=int, default=100, help='num epochs')
cmd_opt.add_argument('-iters_per_eval', type=int, default=100, help='iterations per evaluation')
cmd_opt.add_argument('-flow_layers', type=int, default=10, help='number of flow layers')
cmd_opt.add_argument('-batch_size', type=int, default=128, help='batch size')
cmd_opt.add_argument('-num_particles', type=int, default=256, help='num particles')
cmd_opt.add_argument('-learning_rate', type=float, default=1e-3, help='learning rate')

# args for ode
cmd_opt.add_argument('-feed_context_input', type=eval, help='whether to feed (particles, ob_m) as first layer input',
                     default=True, choices=[True, False])
cmd_opt.add_argument('-layer_cond', type=str, default='ignore', help='intermediate layer conditioning type', 
                     choices=['ignore', 'concat', 'hyper'])
cmd_opt.add_argument('-hyper_dims', type=str, default='64-64', help='hyper network hidden sizes')

cmd_opt.add_argument('-layer_type', type=str, default='ConcatSquashLinear', help='layer type')
cmd_opt.add_argument('-divergence_fn', type=str, default='brute_force', choices=['brute_force', 'approximate'])
cmd_opt.add_argument('-nonlinearity', type=str, default='tanh', choices=NONLINEARITIES)
cmd_opt.add_argument('-rademacher', type=eval, default=False, choices=[True, False])
cmd_opt.add_argument('-time_length', type=float, default=0.5)
cmd_opt.add_argument('-ode_stepsize', type=float, default=0.1, help='ode stepsize')

cmd_opt.add_argument('-time_mode', type=str, default='fixed', choices=['fixed', 'param', 'adaptive'])
cmd_opt.add_argument('-vsmc', type=str, default='mlp', choices=['smc', 'mlp', 'gru', 'lstm'])

cmd_opt.add_argument('-residual', type=eval, default=False)
cmd_opt.add_argument('-solver', type=str, default='dopri5', choices=SOLVERS)
cmd_opt.add_argument('-fwd_solver', type=str, default=None, choices=SOLVERS + [None])
cmd_opt.add_argument('-backwd_solver', type=str, default=None, choices=SOLVERS + [None])
cmd_opt.add_argument('-atol', type=float, default=1e-5)
cmd_opt.add_argument('-rtol', type=float, default=1e-5)
cmd_opt.add_argument('-test_solver', type=str, default=None, choices=SOLVERS + [None])
cmd_opt.add_argument('-test_atol', type=float, default=None)
cmd_opt.add_argument('-test_rtol', type=float, default=None)
cmd_opt.add_argument('-step_size', type=float, default=None, help='Optional fixed step size.')

cmd_opt.add_argument('-batch_norm', type=eval, default=False, choices=[True, False])
cmd_opt.add_argument('-spectral_norm', type=eval, default=False, choices=[True, False])

cmd_opt.add_argument('-stage_len', type=int, default=-1, help='bptt type of training')
cmd_opt.add_argument('-n_stages', type=int, default=-1, help='number of stages in stagewise training')
cmd_opt.add_argument('-num_blocks', type=int, default=1, help='Number of stacked CNFs.')
cmd_opt.add_argument('-f_layers', type=int, default=3, help='Number of layers in f')
cmd_opt.add_argument('-dims', type=str, default='64-64-64')
cmd_opt.add_argument('-time_pred_dims', type=str, default='64-1')
cmd_opt.add_argument('-weight_decay', type=float, default=1e-5)
cmd_opt.add_argument('-hyper_hid_dims', type=str, default='64')

# args for observation conditional ode
cmd_opt.add_argument('-ob_feed', type=str, default='ob', choices=['ob', 'grad_x'])

cmd_opt.add_argument('-merge_method', type=str, default='linear', choices=['linear', 'ignore'])
cmd_opt.add_argument('-stage_dist_metric', type=str, default='ce', choices=['ce', 'mmd'])
cmd_opt.add_argument('-kernel_embed_dims', type=str, default='64')
cmd_opt.add_argument('-kernel_bw', type=float, default=1.0, help='coefficit of kernel bw')
cmd_opt.add_argument('-train_kernel_embed', type=eval, default=False, choices=[True, False])

# args for toy experiment
cmd_opt.add_argument('-gauss_dim', type=int, default=2, help='dimension of gaussian')
cmd_opt.add_argument('-train_samples', type=int, default=10, help='number of training samples')
cmd_opt.add_argument('-prior_mu', type=float, default=0, help='prior mu')
cmd_opt.add_argument('-prior_sigma', type=float, default=1.0, help='prior sigma')
cmd_opt.add_argument('-data_folder', type=str, default=None, help='dataset folder')
cmd_opt.add_argument('-split_idx', type=int, default=1, help='split index')
cmd_opt.add_argument('-l_sigma', type=float, default=1.0, help='likelihood sigma')
cmd_opt.add_argument('-t_sigma', type=float, default=1.0, help='transition sigma')

cmd_opt.add_argument('-num_vals', type=int, default=10, help='evaluate on several val seqs')

# args for rotate lr
cmd_opt.add_argument('-dim_x', type=int, default=0, help='first dim for rotate')
cmd_opt.add_argument('-dim_y', type=int, default=0, help='second dim for rotate')
cmd_opt.add_argument('-max_angle', type=float, default=np.pi/12, help='max angle for rotate')
cmd_opt.add_argument('-max_degree', type=float, default=None, help='max degree for rotate')
cmd_opt.add_argument('-meta_type', type=str, default='single', help='lr meta type', choices=['single', 'angle', 'paired'])

# args for kbr
cmd_opt.add_argument('-num_data', type=int, default=1000, help='number of training samples for kernel MC')

# args for test
cmd_opt.add_argument('-test_epoch', type=int, default=10, help='number of sequences of observations for test')
cmd_opt.add_argument('-test_length', type=int, default=10, help='number of observations for test')
cmd_opt.add_argument('-num_mc_samples', type=int, default=5000, help='number of MC samples to compute true E_p')

cmd_args = cmd_opt.parse_args()

if cmd_args.fwd_solver is None:
    cmd_args.fwd_solver = cmd_args.solver

if cmd_args.backwd_solver is None:
    cmd_args.backwd_solver = cmd_args.solver

if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

if cmd_args.max_degree is not None:
    cmd_args.max_angle = cmd_args.max_degree / 180.0 * np.pi

if cmd_args.init_model_dump == 'None':
    cmd_args.init_model_dump = None
print(cmd_args)
