#!/bin/bash

python3 train_potter-hand-pose-3d.py \
    --pretrained_ckpt output/ego4d/PoolAttnHRCam_Pose_3D/potter_pose_3d_ego4d-auotmatic-twoMLP-filtered/POTTER-HandPose-ego4d-val.pt \
    --anno_type annotation \
    --gpu_number 1