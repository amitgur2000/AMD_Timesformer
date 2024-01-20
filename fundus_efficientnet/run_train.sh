#!/bin/bash
# $1: resume checkpoint. example: checkpoints/model_199.pth

torchrun --nproc_per_node=4 train.py --data-path /DCAIOCTO/ModelDataZoo/datasets/AMD_data/fundus_DB_480/\
 --model efficientnet_v2_l --batch-size 32 --lr 0.1 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5\
 --lr-warmup-method linear --epochs 200 --random-erase 0.1 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0\
 --weight-decay 0.000002 --norm-weight-decay 0.0 --train-crop-size 384 --val-crop-size 480 --val-resize-size 480\
 --ra-reps 3 --weights EfficientNet_V2_L_Weights.DEFAULT --save_every 30 --fine_tune --auto-augment ta_wide\
 --ra-sampler
 #--resume $1
                
#  --model-ema
#--opt adamw  
# 
