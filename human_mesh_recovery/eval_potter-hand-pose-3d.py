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
from hybrik.datasets.generators import testChunkedGenerator


def main(args):
    torch.cuda.empty_cache()
    cfg = update_config(args.cfg_file)
    device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')

    ############ MODEL ###########
    model = PoolAttnHR_Pose_3D(**cfg.MODEL)
    # Load pretrained cls_weight or available hand pose weight
    load_pretrained_weights(model, torch.load(args.pretrained_ckpt, map_location=device))
    print('Loaded pretrained hand pose estimation weight')
    model = model.to(device)
    model.eval()

    ########### DATASET ###########
    # Load Ego4D dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_dataset = ego4dDataset(cfg, 
                                 anno_type=args.anno_type, 
                                 split='test', 
                                 transform=transform,
                                 use_preset=args.use_preset)
    ##################################
    receptive_field = cfg.MODEL.RECEPTIVE_FIELD
    pad = (receptive_field - 1) // 2 # Padding on each side
    test_batch_size = 1
    stride = 1
    ##################################
    # val dataloader
    input_val, gt_pose3d_val, vis_flag_val, hand_wrist_val, _, _, _, _ = test_dataset.get_data_by_subject()
    test_generator = testChunkedGenerator(test_batch_size//stride, input_val, gt_pose3d_val, vis_flag_val, hand_wrist_val,
                                      stride, pad=pad, seq_to_one=True, shuffle=False)
    
    print(f'Number of used takes: {len(test_dataset.curr_split_take)}')

    ######### Inferece #########
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()

    with torch.no_grad():
        for batch_input, batch_3d_gt, batch_vis_flag, batch_hand_wrist in tqdm(test_generator.next_epoch(), total=test_generator.num_batches): # (1,F,3,224,224) (1,1,21,3) (1,1,21)
            # compute output
            input = batch_input.to(device)
            _, pose_3d_pred = model(input) # (1,21,3)

            # Unnormalize predicted and GT pose 3D kpts
            pred_3d_pts = pose_3d_pred.cpu().detach().numpy()
            pred_3d_pts = pred_3d_pts * test_dataset.joint_std + test_dataset.joint_mean
            gt_3d_kpts = batch_3d_gt.squeeze(1).cpu().detach().numpy()
            gt_3d_kpts = gt_3d_kpts * test_dataset.joint_std + test_dataset.joint_mean # (1,21,3)

            # TODO: Add hand wrist in chunk generator   
            B, D = batch_3d_gt.shape[0], batch_3d_gt.shape[-1]
            hand_wrist_kpts = batch_hand_wrist.view(B,1,D).expand(B,21,D)
            hand_wrist_kpts = hand_wrist_kpts.cpu().detach().numpy()
            pred_3d_pts = (pred_3d_pts + hand_wrist_kpts*1000)
            gt_3d_kpts = (gt_3d_kpts + hand_wrist_kpts*1000)

            # Compute MPJPE
            vis_flag = batch_vis_flag.squeeze(1).to(bool)
            valid_pred_3d_kpts = torch.from_numpy(pred_3d_pts)
            valid_pred_3d_kpts = valid_pred_3d_kpts[vis_flag].view(1,-1,3)
            valid_pose_3d_gt = torch.from_numpy(gt_3d_kpts)
            valid_pose_3d_gt = valid_pose_3d_gt[vis_flag].view(1,-1,3)
            epoch_loss_3d_pos.update(mpjpe(valid_pred_3d_kpts, valid_pose_3d_gt).item(), B)
            epoch_loss_3d_pos_procrustes.update(p_mpjpe(valid_pred_3d_kpts, valid_pose_3d_gt), B)

    print(f'MPJPE: {epoch_loss_3d_pos.avg:.2f} (mm)')
    print(f'P-MPJPE: {epoch_loss_3d_pos_procrustes.avg:.2f} (mm)')



if __name__ == '__main__':
    args = parse_args_function()
    main(args)
