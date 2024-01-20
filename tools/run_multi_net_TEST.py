# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import sys, os, shutil, csv
sys.path.append('./')
"""Wrapper to train and test a video classification model."""
from timesformer.utils.misc import launch_job
from timesformer.utils.parser import load_config, parse_args
import timesformer.utils.multiprocessing as mpu
from multiprocessing import Queue

from tools.test_net import test
from tools.train_net import train
import torch
import numpy as np
from scipy.special import softmax


def get_func(cfg):
    train_func = train
    test_func = test
    return train_func, test_func

def lunch_job_new(cfg, init_method, func, daemon=False):
    # out_queue = Queue()
    # torch.multiprocessing.spawn(
    #         mpu.run,
    #         nprocs=cfg.NUM_GPUS,
    #         args=(
    #             cfg.NUM_GPUS,
    #             func,
    #             init_method,
    #             cfg.SHARD_ID,
    #             cfg.NUM_SHARDS,
    #             cfg.DIST_BACKEND,
    #             cfg,
    #             out_queue,    # AG added an output queue 
    #         ),
    #         daemon=daemon,
    # )
    return test(cfg)

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    if args.num_shards > 1:
       args.output_dir = str(args.job_dir)
    cfg = load_config(args)

    train, test = get_func(cfg)

    # Perform multi-clip testing.
    old_CP_path = "/home/agur/projects/AMD/AMD_Timesformer/old_checkpoints/"
    new_CP_path = "/home/agur/projects/AMD/AMD_Timesformer/checkpoints/checkpoint_epoch_00025.pyth"
    checkpoint_files = os.listdir(old_CP_path)
    all_video_files=[]
    with open(cfg.DATA.PATH_TO_DATA_DIR+'test.csv', 'r', newline='') as csvfile:
        video_paths = csv.reader(csvfile, delimiter=' ')
        for row in video_paths:
            all_video_files.append(row)
    if cfg.TEST.ENABLE:
    # AG: add a loop to run multi test on multi checkpoints
        all_video_preds=[]
        for cp_file in checkpoint_files:
            shutil.copy(old_CP_path+cp_file,new_CP_path)
            video_preds, video_labels = lunch_job_new(cfg=cfg, init_method=args.init_method, func=test)
            video_preds_sm = np.array([softmax(pred_score) for pred_score in video_preds])
            all_video_preds.append(video_preds_sm)
        video_pred_avg = sum(all_video_preds)/len(all_video_preds)
        for (vid_label, vid_pred, video_file) in zip(video_labels, video_pred_avg, all_video_files):
            file_name_list = video_file[0].split("/")
            file_name= file_name_list[-2]+"_"+file_name_list[-1]
            print(vid_label, vid_pred, video_file[0], file_name)
            
    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)


if __name__ == "__main__":
    main()
