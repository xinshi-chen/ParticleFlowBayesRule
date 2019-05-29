#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

result_root=../../../dropbox/results

n_train=30
layer_cond=concat
feed_ctx=True
sol=rk4
t=adaptive
np=512
phase=visualize

subdir=n-${n_train}-cond-${layer_cond}-ctx-${feed_ctx}-sol-${sol}-t-${t}-np-${np}

dump_dir=$result_root/two_gaussian/$subdir
out_dir=scratch/$subdir

if [ ! -e $out_dir ];
then
    mkdir -p $out_dir
fi

if [ "$phase" != "train" ];
then
    model_dump=$dump_dir/best_val_model.dump
else
    model_dump=None
fi

seed=1

python main.py \
    -phase $phase \
    -seed $seed \
    -init_model_dump $model_dump \
    -num_epochs 1000 \
    -num_particles $np \
    -feed_context_input $feed_ctx \
    -layer_cond $layer_cond \
    -save_dir $out_dir \
    -gpu 0 \
    -batch_size 1 \
    -solver $sol \
    -train_samples $n_train \
    -train_kernel_embed True \
    -time_mode $t \
    -time_length 1.0 \
    $@
