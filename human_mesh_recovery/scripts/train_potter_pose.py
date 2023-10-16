import argparse
import os
import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPL
from hybrik.utils.render import SMPLRenderer
from hybrik.utils.functions import train, validate
from hybrik.utils.logger import create_logger
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
from hybrik.models.PoolAttnHR_Pose import PoolAttnHR_Pose
from hybrik.datasets.coco import *
from torch.utils.data import Dataset, DataLoader
from hybrik.utils.loss import JointsMSELoss



def main():
    cfg_file = '/home/jinxu/code/POTTER/human_mesh_recovery/configs/potter_pose.yaml'
    CKPT = '/home/jinxu/code/POTTER/human_mesh_recovery/eval/potter_demo.pth'
    cfg = update_config(cfg_file)
    gpus = [int(i) for i in cfg.GPUS.split(',')]
    logger, final_output_dir, tb_log_dir = create_logger(cfg, 
                                                         '/home/jinxu/code/POTTER/human_mesh_recovery/configs/potter_pose.yaml', 
                                                         'train')

    ############ MODEL ###########
    # Define model
    model = PoolAttnHR_Pose(**cfg.MODEL)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    # Load in pretrained POTTER weight
    model_state = model.state_dict()
    pretrained_state = torch.load(CKPT)
    pretrained_state = {k: v for k, v in pretrained_state.items()
                                if k in model_state and v.size() == model_state[k].size()}
    model_state.update(pretrained_state)
    model.load_state_dict(model_state)

    ############# CRITERION AND OPTIMIZER ###########
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.TRAIN.LR
    )

    ########### DATASET ###########
    # Load training dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_dataset = COCODataset(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    # Load validation dataset
    valid_dataset = COCODataset(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE*3,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE*3,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    ############ TRAINING ###########
    best_perf = 0.0
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        # train for one epoch
        logger.info(f'############# Starting Epoch {epoch} #############')
        train(cfg, train_loader, model, criterion, optimizer, epoch, final_output_dir)

        # evaluate on validation set
        perf_indicator = validate(cfg, valid_loader, valid_dataset, model, criterion)

        # Save best model weight
        if perf_indicator > best_perf:
            best_perf = perf_indicator
            # Save model weight
            torch.save(model.state_dict(), os.path.join("/home/jinxu/code/POTTER/human_mesh_recovery/eval" "POTTER_pose_model_best.pth"))


if __name__ == '__main__':
    main()