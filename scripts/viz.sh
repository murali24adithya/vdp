#!/usr/bin/env bash

# This is a script that will evaluate all models for SGCLS
export CUDA_VISIBLE_DEVICES=0

python -O models/_visualize.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 111 -clip 5  -hidden_dim 512 -pooling_dim 4096 -ngpu 1 -ckpt checkpoints/vgrel-motifnet-sgcls.tar -use_bias -limit_vision -val_size 111



