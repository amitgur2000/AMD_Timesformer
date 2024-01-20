#!/bin/bash

# Keep single GPU due to distributed results ...
torchrun --nproc_per_node=1 train.py --data-path /DCAIOCTO/ModelDataZoo/datasets/AMD_data/fundus_DB_480/\
 --model efficientnet_v2_l --batch-size 64 --train-crop-size 384 --val-crop-size 480 --val-resize-size 480\
 --weights EfficientNet_V2_L_Weights.DEFAULT --resume $1\
 --test-only
                
#  --model-ema
#--opt adamw  
# --ra-sampler
