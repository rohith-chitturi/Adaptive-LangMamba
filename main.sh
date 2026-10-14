#!/bin/bash

# Configuration
DATAROOT=${DATAROOT:-"./datasets/Mayo2016_2d"}
GPU_ID=${GPU_ID:--1} # Default to -1 (CPU) for compatibility, change to 0, 1, etc. for GPU

##train
python train.py  --name ME_LEDA   --model meunet   --dataroot "$DATAROOT" --lr 0.0001 --gpu_ids "$GPU_ID" --print_freq 25 --batch_size 1 --lr_policy cosine

##test
python test.py  --name ME_LEDA   --model meunet  --result_name me_results   --gpu_ids "$GPU_ID" --batch_size 1 --eval