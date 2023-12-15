#!/bin/bash

python3 eval_potter-hand-pose-3d.py \
    --pretrained_ckpt output/POTTER_handPose_ego4d_manual+auto.pt \
    --anno_type annotation \
    --gpu_number 0