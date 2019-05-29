from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pfbayes.common.consts import DEVICE
from tqdm import tqdm

LOW = -4
HIGH = 4


def get_flow_heatmaps(flow, mvn_dist, lm_val_gen, val_db, mus):    
    # get particles from each stage of flow
    particles = mvn_dist.get_samples(1024)
    densities = mvn_dist.get_log_pdf(particles)
    list_particles = []
    val_gen = lm_val_gen()
    for ob in val_gen:
        list_particles.append(particles.detach())
        particles, densities = flow(particles, densities, 
                                    prior_samples=particles,
                                    ob_m=ob)

    # reverse flow
    val_gen = lm_val_gen()    
    hist_obs = []
    w = int(np.sqrt(mus.shape[0]))
    list_scores = []
    for t, ob in enumerate(val_gen):
        hist_obs.append(ob)
        particles = mus
        densities = torch.zeros((mus.shape[0], 1), dtype=torch.float32).to(DEVICE)
        for o in reversed(hist_obs):
            particles, densities = flow(particles, densities, 
                                        prior_samples=list_particles[t],
                                        ob_m=o,
                                        reverse=True)
        log_pz = mvn_dist.get_log_pdf(particles)
        log_px = log_pz - densities
        scores = torch.softmax(log_px.view(-1), -1).view(w, w).data.cpu().numpy()
        list_scores.append(scores)
    return list_scores


def get_true_heatmaps(mvn_dist, lm_val_gen, val_db, mus):
    val_gen = lm_val_gen()
    w = int(np.sqrt(mus.shape[0]))
    list_scores = []
    hist_obs = []
    for t, ob in enumerate(val_gen):
        hist_obs.append(ob)
        obs = torch.cat(hist_obs, dim=0)
        log_scores = val_db.log_posterior(mus, obs)
        exact_scores = torch.softmax(log_scores.view(-1), -1).view(w, w).data.cpu().numpy()
        list_scores.append(exact_scores)
    return list_scores


def get_normalized_heatmaps(mvn_dist, lm_val_gen, val_db, mus):
    val_gen = lm_val_gen()
    w = int(np.sqrt(mus.shape[0]))
    list_scores = []
    hist_obs = []
    for t, ob in enumerate(val_gen):
        hist_obs.append(ob)
        obs = torch.cat(hist_obs, dim=0)
        log_prior = val_db.log_prior(mus)
        log_likelihood = torch.tensor([0] * mus.shape[0], dtype=torch.float32).reshape(-1, 1).to(DEVICE)
        for i in tqdm(range(mus.shape[0])):    
            ll_i = val_db.log_likelihood(obs, mus[i, :].reshape(1, -1))
            log_likelihood[i, 0] = torch.sum(ll_i)
        log_pos = log_prior + log_likelihood
        scores = torch.softmax(log_pos.view(-1), -1).view(w, w).cpu().data.numpy()
        list_scores.append(scores)
    return list_scores


def plot_normalized_density(particles, prior_func, ll_func):
    num_particles = particles.shape[0]
    w = int(np.sqrt(num_particles))  # assume plotting on squared image
    assert w ** 2 == num_particles
    log_prior = prior_func(mus)    
    log_likelihood = torch.tensor([0] * num_particles, dtype=torch.float32).reshape(-1, 1)
    for i in range(num_particles): 
        ll_i = ll_func(particles[i, :].reshape(1, -1))
        log_likelihood[i, 0] = torch.sum(ll_i)
    log_pos = log_prior + log_likelihood
    scores = torch.softmax(log_pos.view(-1), -1).view(w, w).numpy()
    plt.subplot(1, 2, 1)
    plt.imshow(scores.reshape((w, w)))
    plt.axis('equal')
    plt.axis('off')


def plt_potential_func(potential, ax, npts=100, title="$p(x)$"):
    """
    Args:
        potential: computes U(z_k) given z_k
    """
    xside = np.linspace(LOW, HIGH, npts)
    yside = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(xside, yside)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    z = torch.Tensor(z)
    u = potential(z).cpu().numpy()
    p = np.exp(-u).reshape(npts, npts)

    plt.pcolormesh(xx, yy, p)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow(prior_logdensity, transform, ax, npts=100, title="$q(x)$", device="cpu"):
    """
    Args:
        transform: computes z_k and log(q_k) given z_0
    """
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    z = torch.tensor(z, requires_grad=True).type(torch.float32).to(device)
    logqz = prior_logdensity(z)
    logqz = torch.sum(logqz, dim=1)[:, None]
    z, logqz = transform(z, logqz)
    logqz = torch.sum(logqz, dim=1)[:, None]

    xx = z[:, 0].cpu().numpy().reshape(npts, npts)
    yy = z[:, 1].cpu().numpy().reshape(npts, npts)
    qz = np.exp(logqz.cpu().numpy()).reshape(npts, npts)

    plt.pcolormesh(xx, yy, qz)
    ax.set_xlim(LOW, HIGH)
    ax.set_ylim(LOW, HIGH)
    cmap = matplotlib.cm.get_cmap(None)
    ax.set_facecolor(cmap(0.))
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow_density(prior_logdensity, inverse_transform, ax, npts=100, memory=100, title="$q(x)$", device="cpu"):
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    x = torch.from_numpy(x).type(torch.float32).to(device)
    zeros = torch.zeros(x.shape[0], 1).to(x)

    z, delta_logp = [], []
    inds = torch.arange(0, x.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory**2)):
        z_, delta_logp_ = inverse_transform(x[ii], zeros[ii])
        z.append(z_)
        delta_logp.append(delta_logp_)
    z = torch.cat(z, 0)
    delta_logp = torch.cat(delta_logp, 0)

    logpz = prior_logdensity(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    px = np.exp(logpx.cpu().numpy()).reshape(npts, npts)

    ax.imshow(px)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow_samples(prior_sample, transform, ax, npts=100, memory=100, title="$x ~ q(x)$", device="cpu"):
    z = prior_sample(npts * npts, 2).type(torch.float32).to(device)
    zk = []
    inds = torch.arange(0, z.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory**2)):
        zk.append(transform(z[ii]))
    zk = torch.cat(zk, 0).cpu().numpy()
    ax.hist2d(zk[:, 0], zk[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_samples(samples, ax, npts=100, title="$x ~ p(x)$"):    
    ax.hist2d(samples[:, 0], samples[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def visualize_transform(
    potential_or_samples, prior_sample, prior_density, transform=None, inverse_transform=None, samples=True, npts=100,
    memory=100, device="cpu"
):
    """Produces visualization for the model density and samples from the model."""
    plt.clf()
    ax = plt.subplot(1, 3, 1, aspect="equal")
    if samples:
        plt_samples(potential_or_samples, ax, npts=npts)
    else:
        plt_potential_func(potential_or_samples, ax, npts=npts)

    ax = plt.subplot(1, 3, 2, aspect="equal")
    if inverse_transform is None:
        plt_flow(prior_density, transform, ax, npts=npts, device=device)
    else:
        plt_flow_density(prior_density, inverse_transform, ax, npts=npts, memory=memory, device=device)

    ax = plt.subplot(1, 3, 3, aspect="equal")
    if transform is not None:
        plt_flow_samples(prior_sample, transform, ax, npts=npts, memory=memory, device=device)