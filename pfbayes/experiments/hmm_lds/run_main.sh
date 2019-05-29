#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

result_root=../../../dropbox/results

n_train=25
layer_cond=concat
feed_context=True
stage_len=-1
solver=rk4
num_particles=1024
eval_particles=$num_particles
phase=eval_metric
dim=2
subdir=n-${n_train}-cond-${layer_cond}-ctx-${feed_context}-sl-${stage_len}-sol-${solver}-np-${num_particles}-dim-${dim}

dump_dir=$result_root/hmm_lds/$subdir
out_dir=scratch/$subdir

if [ ! -e $out_dir ];
then
    mkdir -p $out_dir
fi

if [ "$phase" != "train" ];
then
    model_dump=$dump_dir/best_val_model.dump
    gpu=-1
else
    model_dump=None
    gpu=0
fi

python main.py \
    -num_epochs 1000 \
    -gpu $gpu \
    -num_particles $eval_particles \
    -stage_len $stage_len \
    -batch_size 1 \
    -phase $phase \
    -init_model_dump $model_dump \
    -feed_context_input $feed_context \
    -layer_cond $layer_cond \
    -gauss_dim $dim \
    -train_samples $n_train \
    -train_kernel_embed True \
    -save_dir $out_dir \
    -solver $solver \
    -time_mode adaptive \
    -time_length 1.0 \
    -test_length $n_train \
    $@

