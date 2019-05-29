from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import os
import torch
import random
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
import numpy as np
from pfbayes.common.data_utils.dataset import RealSupervisedDataset
from pfbayes.common.distributions import DiagMvn
from pfbayes.common.consts import DEVICE, t_float
from tqdm import tqdm


def _shuffle(x, y):
    assert x.shape[0] == y.shape[0]
    assert y.shape[1] == 1
    t = np.concatenate([x, y], axis=-1)
    np.random.shuffle(t)
    x = t[:, :-1]
    y = t[:, -1].reshape(-1, 1)
    return x, y


class MnistDataset(RealSupervisedDataset):
           
    def __init__(self, data_folder):
        super(MnistDataset, self).__init__()
        xtrain = np.load(os.path.join(data_folder, 'xtrain.npy'))
        ytrain = np.load(os.path.join(data_folder, 'ytrain.npy')).astype(np.float32)
        xtest = np.load(os.path.join(data_folder, 'xtest.npy'))
        ytest = np.load(os.path.join(data_folder, 'ytest.npy')).astype(np.float32)

        xtrain, ytrain = _shuffle(xtrain, ytrain)
        xtest, ytest = _shuffle(xtest, ytest)

        n = xtest.shape[0]
        xval, yval = xtrain[:n, :], ytrain[:n, :]
        xtrain, ytrain = xtrain[n:, :], ytrain[n:, :]

        xtrain = torch.tensor(xtrain)
        ytrain = torch.tensor(ytrain)
        self.data_partitions['train'] = (xtrain, ytrain)

        xval = torch.tensor(xval)
        yval = torch.tensor(yval)
        self.data_partitions['val'] = (xval, yval)

        xtest = torch.tensor(xtest)
        ytest = torch.tensor(ytest, dtype=t_float)
        self.data_partitions['test'] = (xtest, ytest)

        self.x_dim = xtrain.shape[1]
        self.ob_dim = xtrain.shape[1] + ytrain.shape[1]
        self.num_train = xtrain.shape[0]
        self.num_test = xtest.shape[0]

        print('num_train:', xtrain.shape[0])
        print('num_val:', xval.shape[0])
        print('num_test:', xtest.shape[0])

    def log_likelihood(self, obs, x):
        feats, labels = obs
        pred = torch.matmul(feats, x.t())
        labels = labels.repeat(1, x.shape[0])
        ll = labels * F.logsigmoid(pred) + (1 - labels) * F.logsigmoid(-pred)
        return ll


class RotateMnistDataset(object):

    def _rand_angle(self):
        return np.random.rand() * 2 * self.max_angle - self.max_angle

    def _rotate(self, mat, angle):
        rotate_mat = torch.tensor([[np.cos(angle), np.sin(angle)],
                                   [-np.sin(angle), np.cos(angle)]], dtype=t_float).to(mat)
        sl1, sl2 = mat[:, self.dim_x].view(-1, 1), mat[:, self.dim_y].view(-1, 1)
        sl = torch.cat([sl1, sl2], dim=-1)
        rotated = torch.matmul(sl, rotate_mat)
        mat[:, self.dim_x] = rotated[:, 0]
        mat[:, self.dim_y] = rotated[:, 1]
        return mat

    def log_likelihood(self, obs, x):
        feats, labels = obs
        pred = torch.matmul(feats, x.t())
        labels = labels.repeat(1, x.shape[0])
        ll = labels * F.logsigmoid(pred) + (1 - labels) * F.logsigmoid(-pred)
        return ll

    def __init__(self, data_folder, num_vals, max_angle, dim_x, dim_y):
        super(RotateMnistDataset, self).__init__()
        self.data_partitions = {}
        xtrain = np.load(os.path.join(data_folder, 'xtrain.npy'))
        ytrain = np.load(os.path.join(data_folder, 'ytrain.npy')).astype(np.float32)
        xtest = np.load(os.path.join(data_folder, 'xtest.npy'))
        ytest = np.load(os.path.join(data_folder, 'ytest.npy')).astype(np.float32)
        self.max_angle = max_angle
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.angles = {}
        self.angles['test'] = self._rand_angle()

        xtrain, ytrain = _shuffle(xtrain, ytrain)
        xtest, ytest = _shuffle(xtest, ytest)

        n = xtest.shape[0]
        offset = 0
        for i in range(num_vals):
            xval, yval = xtrain[offset:offset+n, :], ytrain[offset:offset+n, :]
            xval = torch.tensor(xval)
            yval = torch.tensor(yval)
            angle = self._rand_angle()
            xval = self._rotate(xval, angle)
            self.angles['val-%d' % i] = angle
            self.data_partitions['val-%d' % i] = (xval, yval)
            offset += n
        xtrain, ytrain = xtrain[offset:, :], ytrain[offset:, :]

        xtrain = torch.tensor(xtrain)
        ytrain = torch.tensor(ytrain)
        self.data_partitions['train'] = (xtrain, ytrain)

        xtest = torch.tensor(xtest)
        ytest = torch.tensor(ytest, dtype=t_float)
        self.data_partitions['test'] = (xtest, ytest)

        self.x_dim = xtrain.shape[1]
        self.ob_dim = xtrain.shape[1] + ytrain.shape[1]
        self.num_train = xtrain.shape[0]
        self.num_test = xtest.shape[0]

        print('num_train:', xtrain.shape[0])
        print('num_test:', xtest.shape[0])
        print(self.angles)
        xtest, ytest = self.data_partitions['test']
        xtest = self._rotate(xtest, self.angles['test'])
        self.data_partitions['test'] = (xtest, ytest)

    def data_gen(self, batch_size, phase, auto_reset=True, shuffle=True):
        assert phase in self.data_partitions

        feats, labels = self.data_partitions[phase]
        num_obs = feats.shape[0]
        if not phase in self.angles:
            angle = self._rand_angle()
            print(angle)
        else:
            angle = None
        while True:
            if shuffle:
                perms = torch.randperm(num_obs)
                buf_feats = feats[perms, :]
                buf_labels = labels[perms, :]
            else:
                buf_feats = feats
                buf_labels = labels
            for pos in range(0, num_obs, batch_size):
                if pos + batch_size > num_obs:  # the last mini-batch has fewer samples
                    if auto_reset:  # no need to use this last mini-batch
                        break
                    else:
                        num_samples = num_obs - pos
                else:
                    num_samples = batch_size
                feat = buf_feats[pos : pos + num_samples, :].to(DEVICE)
                if angle is not None:
                    feat = self._rotate(feat, angle)
                yield feat, buf_labels[pos : pos + num_samples, :].to(DEVICE)
            if not auto_reset:
                break


def eval_particles_acc(particles, val_gen):
    all_labels = []
    all_preds = []
    for ob in val_gen:
        feats, labels = ob
        pred = torch.sigmoid(torch.matmul(feats, particles.t()))
        all_preds.append(pred)
        all_labels.append(labels)
    labels = torch.cat(all_labels, dim=0).repeat(1, particles.shape[0])
    preds = torch.cat(all_preds, dim=0)
    acc = (labels < 0.5) == (preds < 0.5)
    acc = torch.mean(torch.mean(acc.float(), dim=0))    
    return acc.item()
