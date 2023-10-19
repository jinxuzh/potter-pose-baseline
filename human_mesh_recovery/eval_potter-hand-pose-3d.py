import argparse
import os
import cv2
import numpy as np
import torch
import time
from easydict import EasyDict as edict
from hybrik.utils.config import update_config
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
from hybrik.models.PoolAttnHR_Pose_3D import PoolAttnHR_Pose_3D, load_pretrained_weights
from hybrik.utils.loss import *
from hybrik.utils.functions import AverageMeter
from hybrik.utils.vis import save_debug_images
from hybrik.utils.evaluate import accuracy
from hybrik.datasets.ego4d_dataset import ego4dDataset
from hybrik.utils.option import parse_args_function


def main(args):
    torch.cuda.empty_cache()
    cfg = update_config(args.cfg_file)
    device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')

    ############ MODEL ###########
    model = PoolAttnHR_Pose_3D(**cfg.MODEL)
    # Load pretrained cls_weight or available hand pose weight
    load_pretrained_weights(model, torch.load(args.pretrained_ckpt))
    print('Loaded pretrained hand pose estimation weight')
    model = model.to(device)

    ########### DATASET ###########
    # Load Ego4D dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    valid_dataset = ego4dDataset(cfg, 
                                 anno_type=args.anno_type, 
                                 split='test', 
                                 transform=transform,
                                 use_preset=args.use_preset)
    # Dataloader
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    print(f'Number of takes: Train: {len(valid_dataset.curr_split_take)}\t Val: {len(valid_dataset.curr_split_take)}')

    ######### Inferece #########
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()

    with torch.no_grad():
        valid_loader = tqdm(valid_loader, dynamic_ncols=True)
        for i, (input, _, _, _, pose_3d_gt, vis_flag, meta) in enumerate(valid_loader):
            # Pose 3D prediction
            input = input.to(device)
            _, pose_3d_pred = model(input)

            # Unnormalize predicted and GT pose 3D kpts
            pred_3d_pts = pose_3d_pred.cpu().detach().numpy()
            pred_3d_pts = pred_3d_pts * valid_dataset.joint_std + valid_dataset.joint_mean
            gt_3d_kpts = pose_3d_gt.cpu().detach().numpy()
            gt_3d_kpts = gt_3d_kpts * valid_dataset.joint_std + valid_dataset.joint_mean
            # Add hand wrist
            B, D = meta['hand_wrist'].shape
            hand_wrist_kpts = meta['hand_wrist'].view(B,1,D).expand(B,21,D)
            hand_wrist_kpts = hand_wrist_kpts.cpu().detach().numpy()
            pred_3d_pts = (pred_3d_pts + hand_wrist_kpts*1000)
            gt_3d_kpts = (gt_3d_kpts + hand_wrist_kpts*1000)

            # Compute MPJPE
            valid_pred_3d_kpts = torch.from_numpy(pred_3d_pts)
            valid_pred_3d_kpts[~vis_flag] = 0
            valid_pose_3d_gt = torch.from_numpy(gt_3d_kpts)
            valid_pose_3d_gt[~vis_flag] = 0
            num_valid = torch.sum(vis_flag).item()
            epoch_loss_3d_pos.update(mpjpe(valid_pred_3d_kpts, valid_pose_3d_gt, num_valid).item(), B)
            epoch_loss_3d_pos_procrustes.update(p_mpjpe(valid_pred_3d_kpts, valid_pose_3d_gt, num_valid), B)

    print(f'Protocol #1   (MPJPE) action-wise average: {epoch_loss_3d_pos.avg:.2f} (mm)')
    print(f'Protocol #2 (P-MPJPE) action-wise average: {epoch_loss_3d_pos_procrustes.avg:.2f} (mm)')



if __name__ == '__main__':
    args = parse_args_function()
    main(args)
