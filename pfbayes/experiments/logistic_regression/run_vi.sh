#!/bin/bash

dropbox=../../../dropbox
data_dir=$dropbox/data/mnist
result_root=$dropbox/results

export CUDA_VISIBLE_DEVICES=0

for seed in `seq 1 10`; do

layer_cond=ignore
feed_ctx=True
n_particles=1000
bsize=50
n_stages=10
st_len=2
solver=dopri5
dist_metric=mmd
t=fixed

subdir=ns-${n_stages}-sl-${st_len}-b-${bsize}-sol-${solver}-dm-${dist_metric}-t-${t}-seed-$seed
out_dir=scratch/$subdir

if [ ! -e $out_dir ];
then
    mkdir -p $out_dir
fi

python main_lr_vi.py \
    -num_epochs 70 \
    -seed $seed \
    -stage_len $st_len \
    -dims 64-64 \
    -stage_dist_metric $dist_metric \
    -iters_per_eval $n_stages \
    -num_particles $n_particles \
    -feed_context_input $feed_ctx \
    -layer_cond $layer_cond \
    -save_dir $out_dir \
    -data_folder $data_dir \
    -n_stages $n_stages \
    -gpu 0 \
    -batch_size $bsize \
    -train_kernel_embed True \
    -time_mode $t \
    -time_length 1.0 \
    -solver $solver \
    $@

done
