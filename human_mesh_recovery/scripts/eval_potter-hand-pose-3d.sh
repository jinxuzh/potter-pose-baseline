#!/bin/bash

python3 eval_potter-hand-pose-3d.py \
    --pretrained_ckpt output/ego4d/PoolAttnHRCam_Pose_3D/potter_pose_3d_ego4d-finetune-84takes-2ndAuto/POTTER-HandPose-ego4d-val.pt \
    --anno_type annotation \
    --gpu_number 0