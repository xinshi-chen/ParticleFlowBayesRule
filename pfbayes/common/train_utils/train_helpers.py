from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.optim as optim
from itertools import chain
import os
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pfbayes.common.distributions import KDE, MMD


def all_ll(hist_obs, x, func_ll):
    ll = 0.0
    for ob in hist_obs:
        ll = ll + func_ll(ob, x)
    return ll


def _forward(flow, particles, densities, data_gen, func_ll, func_log_prior, max_steps=-1):
    loss = None
    hist_obs = []
    for t, ob in enumerate(data_gen):
        if loss is None:
            loss = 0.0
        # evolve the particles
        new_particles, new_densities = flow(particles, densities, 
                                            prior_samples=particles,
                                            ob_m=ob)
        hist_obs.append(ob)
        # log likelihood
        ll = func_ll(hist_obs, new_particles)
        log_prior = func_log_prior(new_particles, particles)
        loss_t = torch.mean(new_densities) - torch.mean(ll) - torch.mean(log_prior)
        loss += loss_t
        particles = new_particles
        densities = new_densities
        if max_steps > 0 and t + 1 >= max_steps:
            break
    return loss, particles.detach(), densities.detach()


def forward_global_x(flow, particles, densities, data_gen, func_ll, func_log_prior, max_steps=-1):
    t_func_ll = lambda a, b: all_ll(a, b, func_ll)
    t_log_prior = lambda new_x, old_x: func_log_prior(new_x)
    return _forward(flow, particles, densities, data_gen, t_func_ll, t_log_prior, max_steps)


def forward_local_x(flow, particles, densities, data_gen, func_ll, func_log_transit, max_steps=-1):
    t_func_ll = lambda a, b: func_ll(a[-1], b)
    return _forward(flow, particles, densities, data_gen, t_func_ll, func_log_transit, max_steps)


def _train_loop(cmd_args, get_init_dist, flow, optimizer,
                func_ll, update_prior_func, eval_func):
    best_val_loss = None
    for epoch in range(cmd_args.num_epochs):
        pbar = tqdm(range(cmd_args.iters_per_eval))
        for it in pbar:
            train_gen, prior_dist, func_log_prior = get_init_dist(cmd_args.batch_size)
            particles = prior_dist.get_samples(cmd_args.num_particles)
            densities = prior_dist.get_log_pdf(particles)

            loss_list = []
            cur_func_prior = func_log_prior
            while True:
                optimizer.zero_grad()
                loss, particles, densities = _forward(flow, particles, densities, train_gen,
                                                      func_ll, cur_func_prior, max_steps=cmd_args.stage_len)
                if loss is None:  # no data left for training
                    break
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                if cmd_args.n_stages > 0:  # we need to train next stage
                    cur_func_prior = update_prior_func(particles)
                else:
                    break
            loss = np.sum(loss_list)
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + float(it + 1) / cmd_args.iters_per_eval, loss))

        loss = eval_func(flow, prior_dist)
        if best_val_loss is None or loss < best_val_loss:
            best_val_loss = loss
            print('saving model with best valid error')
            torch.save(flow.state_dict(), os.path.join(cmd_args.save_dir, 'best_val_model.dump'))


def _kde_prior_func(particles):
    kde = KDE(particles)
    cur_func_prior = lambda x, y: kde.log_pdf(x)
    return cur_func_prior


def seg_train_global_x_loop(cmd_args, get_init_dist, flow, optimizer,
                            func_ll, eval_func):
    t_func_ll = lambda a, b: all_ll(a, b, func_ll)
    return _train_loop(cmd_args, get_init_dist, flow, optimizer,
                        t_func_ll, None, eval_func)


def train_global_x_loop(cmd_args, lm_train_gen, prior_dist, flow, optimizer,
                        func_ll, func_log_prior, eval_func):
    t_func_ll = lambda a, b: all_ll(a, b, func_ll)
    t_log_prior = lambda new_x, old_x: func_log_prior(new_x)
    get_init_dist = lambda bsize: (lm_train_gen(bsize), prior_dist, t_log_prior)
    return _train_loop(cmd_args, get_init_dist, flow, optimizer,
                       t_func_ll, _kde_prior_func, eval_func)


def train_local_x_loop(cmd_args, lm_train_gen, prior_dist, flow, optimizer,
                       func_ll, func_log_transit, eval_func):
    t_func_ll = lambda a, b: func_ll(a[-1], b)
    update_func = lambda x: func_log_transit
    get_init_dist = lambda bsize: (lm_train_gen(bsize), prior_dist, func_log_transit)
    return _train_loop(cmd_args, get_init_dist, flow, optimizer,
                       t_func_ll, update_func, eval_func)


def supervised_train_loop(cmd_args, db, prior_dist, flow, ob_net, coeff, eval_func, test_locs=[]):
    print('coeff:', coeff)
    optimizer = optim.Adam(chain(flow.parameters(), ob_net.parameters()),
                           lr=cmd_args.learning_rate,
                           weight_decay=cmd_args.weight_decay)

    best_val_loss = None
    if cmd_args.stage_dist_metric == 'ce':
        fn_log_prior = lambda x, y: KDE(y, coeff=cmd_args.kernel_bw).log_pdf(x)
    else:
        fn_log_prior = lambda x, y: -MMD(x, y, bandwidth=cmd_args.kernel_bw)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, min_lr=1e-6, verbose=True)
    num_obs = 0
    for epoch in range(cmd_args.num_epochs):
        train_gen = db.data_gen(batch_size=cmd_args.batch_size,
                                phase='train',
                                auto_reset=True,
                                shuffle=True)
        pbar = tqdm(range(cmd_args.n_stages))
        particles = prior_dist.get_samples(cmd_args.num_particles)
        densities = prior_dist.get_log_pdf(particles)        
        for it in pbar:
            loss = 0.0
            feats_all = []
            labels_all = []
            particles = particles.detach()
            densities = densities.detach()
            prior_particles = particles 
            optimizer.zero_grad()
            acc = 0.0
            for l in range(cmd_args.stage_len):
                feats, labels = next(train_gen)
                num_obs += feats.shape[0]                
                ob = ob_net((feats, labels))
                if l + 1 == cmd_args.stage_len:
                    pred = torch.sigmoid(torch.matmul(feats, particles.t()))
                    pred = torch.mean(pred, dim=-1, keepdim=True)
                    acc = (labels < 0.5) == (pred < 0.5)
                    acc = torch.mean(acc.float())
                new_particles, new_densities = flow(particles, densities,
                                                    prior_samples=particles,
                                                    ob_m=ob)
                feats_all.append(feats)
                labels_all.append(labels)
                feats = torch.cat(feats_all, dim=0)
                labels = torch.cat(labels_all, dim=0)

                ll = torch.mean(torch.sum(db.log_likelihood((feats, labels), new_particles), dim=0))
                if it == 0:
                    log_prior = prior_dist.get_log_pdf(new_particles) * coeff
                else:
                    log_prior = fn_log_prior(new_particles, prior_particles)
                loss += coeff * torch.mean(new_densities) - ll - torch.mean(log_prior)
                particles = new_particles
                densities = new_densities
            loss.backward()
            optimizer.step()
            if len(test_locs) and num_obs >= test_locs[0]:
                eval_func(num_obs, particles, db, 'test')
                test_locs = test_locs[1:]
            pbar.set_description('epoch %.2f, loss: %.4f, last_acc: %.4f' % (epoch + float(it + 1) / cmd_args.n_stages, loss.item(), acc))
        if (epoch + 1) * cmd_args.n_stages % cmd_args.iters_per_eval == 0:
            loss = 0.0
            if cmd_args.num_vals == 1:
                loss += eval_func(num_obs, particles, db, 'val')
            else:
                for i in range(cmd_args.num_vals):
                    print('evaluating val-%d' % i)
                    loss += eval_func(num_obs, particles, db, 'val-%d' % i)
            loss /= cmd_args.num_vals
            scheduler.step(loss)
            eval_func(num_obs, particles, db, 'test')
            if best_val_loss is None or loss < best_val_loss:
                best_val_loss = loss
                print('saving model with best valid error')
                torch.save(flow.state_dict(), os.path.join(cmd_args.save_dir, 'best_val_model.dump'))
