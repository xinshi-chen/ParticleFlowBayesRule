#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

result_root=../../../dropbox/results

n_train=100
layer_cond=concat
feed_context=True
stage_len=10
sol=rk4
l_sigma=3

# choose from seg_train or eval_metric
phase=eval_metric
dim=2

subdir=n-${n_train}-cond-${layer_cond}-ctx-${feed_context}-sl-${stage_len}-sol-${sol}-ls-${l_sigma}-dim-${dim}

dump_dir=$result_root/mvn_unimodal/$subdir
out_dir=scratch/$subdir

if [ ! -e $out_dir ];
then
    mkdir -p $out_dir
fi

if [[ "$phase" != *"train"* ]];
then
    model_dump=$dump_dir/best_val_model.dump
    gpu=-1
else
    model_dump=None
    gpu=0
fi

python main.py \
    -init_model_dump $model_dump \
    -phase $phase \
    -num_epochs 1000 \
    -l_sigma $l_sigma \
    -gpu 0 \
    -stage_len $stage_len \
    -batch_size 1 \
    -feed_context_input $feed_context \
    -layer_cond $layer_cond \
    -gauss_dim $dim \
    -train_samples $n_train \
    -train_kernel_embed True \
    -save_dir $out_dir \
    -solver $sol \
    -time_mode adaptive \
    -time_length 1.0 \
    -n_stages -1 \
    -iters_per_eval 200 \
    -test_length $n_train \
    $@

