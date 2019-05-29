from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
from pfbayes.common.consts import DEVICE


class ToyDataset(object):

    def __init__(self, partition_sizes={}):
        '''
        Args:
            partition_sizes: dict of str -> int, name and size of each partition
        '''
        self.static_data = {}    
        for key in partition_sizes:
            num_obs = partition_sizes[key]
            self.static_data[key] = self.gen_batch_obs(num_obs).to(DEVICE)

    def gen_batch_obs(self, num_obs):
        raise NotImplementedError

    def _reset(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        self._reset(*args, **kwargs)
        for key in self.static_data:
            num_obs = self.static_data[key].shape[0]
            self.static_data[key] = self.gen_batch_obs(num_obs).to(DEVICE)

    def data_gen(self, batch_size, phase, auto_reset=True, shuffle=True):
        if phase != 'train':
            assert not auto_reset
            assert phase in self.static_data

        if phase in self.static_data:  # generate mini-batches from dataset
            data_split = self.static_data[phase]
            num_obs = data_split.shape[0]
            while True:
                if shuffle:
                    perms = torch.randperm(num_obs)
                    buf_train = data_split[perms, :]
                else:
                    buf_train = data_split
                for pos in range(0, num_obs, batch_size):
                    if pos + batch_size > num_obs:  # the last mini-batch has fewer samples
                        if auto_reset:  # no need to use this last mini-batch
                            break
                        else:
                            num_samples = num_obs - pos
                    else:
                        num_samples = batch_size
                    yield buf_train[pos : pos + num_samples, :]
                if not auto_reset:
                    break
        else:
            while True:
                yield self.gen_batch_obs(batch_size).to(DEVICE)


class RealSupervisedDataset(object):

    def __init__(self):
        self.data_partitions = {}

    def data_gen(self, batch_size, phase, auto_reset=True, shuffle=True):
        assert phase in self.data_partitions

        feats, labels = self.data_partitions[phase]
        num_obs = feats.shape[0]
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
                yield buf_feats[pos : pos + num_samples, :].to(DEVICE), buf_labels[pos : pos + num_samples, :].to(DEVICE)
            if not auto_reset:
                break
