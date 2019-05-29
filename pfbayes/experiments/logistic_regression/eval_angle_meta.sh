#!/bin/bash

dropbox=../../../dropbox
data_dir=$dropbox/data/mnist
result_root=$dropbox/results

export CUDA_VISIBLE_DEVICES=0

layer_cond=ignore
feed_ctx=True
n_particles=256
bsize=32
n_stages=20
solver=rk4
dist_metric=mmd
t=fixed
max_degree=15
dim_x=0
num_vals=10
dim_y=1
phase=test


subdir=ns-${n_stages}-b-${bsize}-sol-${solver}-dm-${dist_metric}-t-${t}-dx-${dim_x}-dy-${dim_y}-md-${max_degree}

dump_dir=$result_root/lr_meta_angle/$subdir
out_dir=scratch/$subdir

if [ ! -e $out_dir ];
then
    mkdir -p $out_dir
fi


for seed in 10 20 30 40 50 60 70 80 90 100; do

python main_lr_meta.py \
    -num_epochs 1000 \
    -max_degree $max_degree \
    -stage_dist_metric $dist_metric \
    -meta_type angle \
    -dim_x $dim_x \
    -dim_y $dim_y \
    -seed $seed \
    -num_particles $n_particles \
    -feed_context_input $feed_ctx \
    -layer_cond $layer_cond \
    -iters_per_eval $n_stages \
    -save_dir $out_dir \
    -data_folder $data_dir \
    -n_stages $n_stages \
    -gpu 0 \
    -batch_size $bsize \
    -train_kernel_embed True \
    -time_mode $t \
    -time_length 1.0 \
    -solver $solver \
    -init_model_dump $dump_dir/best_val_model.dump \
    -phase test \
    $@

done

