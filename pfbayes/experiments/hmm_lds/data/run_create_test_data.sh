#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

n_train=25  # length of sequence is the same as training
epoch=10 # num of sequences for test

python3 create_test_data.py \
    -gauss_dim 10 \
    -threshold 0.7 \
    -train_samples $n_train \
    -test_length $n_train\
    -test_epoch $epoch\
    -save_dir . \
    -seed 101123\
    $@

