#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

n_train=100  # length of sequence is the same as training
epoch=25 # num of sequences for test
l_sigma=3.0  # std of likelihood
dim=3

python3 create_test_data.py \
    -gauss_dim $dim \
    -l_sigma $l_sigma \
    -test_length $n_train \
    -test_epoch $epoch \
    -save_dir . \
    $@

