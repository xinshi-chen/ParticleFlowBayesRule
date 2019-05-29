#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

n_train=25  # length of sequence is the same as training
epoch=25 # num of sequences for test

python3 saved_lds.py \
    -gauss_dim 10 \
    -threshold 0.7 \
    -train_samples $n_train \
    -test_epoch $epoch\
    -seed 1101123\
    $@

