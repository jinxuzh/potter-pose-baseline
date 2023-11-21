import argparse
import os
import cv2
import numpy as np
import torch
import time
from easydict import EasyDict as edict
from hybrik.utils.config import update_config
from hybrik.utils.logger import create_logger
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from hybrik.models.PoolAttnHR_Pose_3D import PoolAttnHR_Pose_3D, load_pretrained_weights
from torch.utils.data import Dataset, DataLoader
from hybrik.utils.loss import Pose3DLoss
from hybrik.utils.functions import AverageMeter
from hybrik.utils.vis import save_debug_images
from hybrik.utils.evaluate import accuracy
from hybrik.datasets.ego4d_dataset import ego4dDataset
from hybrik.utils.option import parse_args_function
from hybrik.datasets.generators import ChunkedGenerator


def train(config, train_loader, model, criterion, optimizer, epoch, output_dir, device, logger, writer_dict):
    total_loss = AverageMeter()

    # switch to train mode
    model.train()
    print_interval = train_loader.num_batches // config.TRAIN_PRINT_NUM

    for i, (batch_input, batch_3d_gt, batch_vis_flag) in tqdm(enumerate(train_loader.next_epoch()), 
                                                              total=train_loader.num_batches):
        # compute output
        input = batch_input.to(device)
        _, pose_3d_pred = model(input)

        # Assign None kpts as zero
        batch_3d_gt[~batch_vis_flag.to(bool)] = 0
        batch_3d_gt = batch_3d_gt.to(device)
        batch_vis_flag = batch_vis_flag.to(device)

        pose_3d_loss = criterion(pose_3d_pred, batch_3d_gt.squeeze(1), batch_vis_flag.squeeze(1))
        loss = pose_3d_loss #hm_loss + pose_3d_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        total_loss.update(loss.item())

        # Log info
        if (i+1) % print_interval == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Total loss {total_loss.val:.5f} ({total_loss.avg:.5f})'.format(
                      epoch, i+1, train_loader.num_batches, total_loss=total_loss)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('Loss/train', total_loss.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

    return total_loss.avg


def validate(config, val_loader, model, criterion, output_dir, device, logger, writer_dict):
    total_loss = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (batch_input, batch_3d_gt, batch_vis_flag) in tqdm(enumerate(val_loader.next_epoch()),
                                                                  total=val_loader.num_batches):
            # compute output
            input = batch_input.to(device)
            _, pose_3d_pred = model(input)

            # Assign None kpts as zero
            batch_3d_gt[~batch_vis_flag.to(bool)] = 0
            batch_3d_gt = batch_3d_gt.to(device)
            batch_vis_flag = batch_vis_flag.to(device)

            pose_3d_loss = criterion(pose_3d_pred, batch_3d_gt.squeeze(1), batch_vis_flag.squeeze(1))
            loss = pose_3d_loss #hm_loss + pose_3d_loss

            # measure accuracy and record loss
            total_loss.update(loss.item())

        # Log info
        msg = 'Test: [{0}/{1}]\t' \
              'Total loss {total_loss.val:.5f} ({total_loss.avg:.5f})'.format(
                    i+1, val_loader.num_batches, total_loss=total_loss)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('Loss/val', total_loss.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return total_loss.avg


def main(args):
    torch.cuda.empty_cache()
    cfg = update_config(args.cfg_file)
    pretrained_hand_pose_CKPT = args.pretrained_ckpt
    device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg_file, 'train')

    ############ MODEL ###########
    model = PoolAttnHR_Pose_3D(**cfg.MODEL)
    # Load pretrained cls_weight or available hand pose weight
    cls_weight = torch.load(args.cls_ckpt)
    if pretrained_hand_pose_CKPT:
        load_pretrained_weights(model, torch.load(pretrained_hand_pose_CKPT, map_location=device))
        logger.info(f'Loaded pretrained weight from {pretrained_hand_pose_CKPT}')
    else:
        load_pretrained_weights(model.poolattnformer_pose.poolattn_cls, cls_weight)
        logger.info('Loaded pretrained POTTER-cls weight')
    model = model.to(device)
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    ############# CRITERION AND OPTIMIZER ###########
    # define loss function (criterion) and optimizer
    criterion = Pose3DLoss().cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.TRAIN.LR
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR
    )

    ########### DATASET ###########
    # Load Ego4D dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ego4dDataset(cfg, 
                                 anno_type=args.anno_type, 
                                 split='train', 
                                 transform=transform,
                                 use_preset=args.use_preset)
    valid_dataset = ego4dDataset(cfg, 
                                 anno_type=args.anno_type, 
                                 split='test', 
                                 transform=transform,
                                 use_preset=args.use_preset)

    ##################################
    receptive_field = cfg.MODEL.RECEPTIVE_FIELD
    pad = (receptive_field - 1) // 2 # Padding on each side
    train_batch_size = cfg.TRAIN.BATCH_SIZE
    val_batch_size = cfg.TEST.BATCH_SIZE
    stride = 1
    ##################################
    # train dataloader
    input_train, gt_pose3d_train, vis_flag_train, _, _, _, _, _ = train_dataset.get_data_by_subject()
    train_generator = ChunkedGenerator(train_batch_size//stride, input_train, gt_pose3d_train, vis_flag_train, stride,
                                       pad=pad, seq_to_one=True, shuffle=True)
    # val dataloader
    input_val, gt_pose3d_val, vis_flag_val, _, _, _, _, _ = valid_dataset.get_data_by_subject()
    val_generator = ChunkedGenerator(val_batch_size//stride, input_val, gt_pose3d_val, vis_flag_val, stride,
                                     pad=pad, seq_to_one=True, shuffle=False)
    
    logger.info(f'Number of takes: Train: {len(train_dataset.curr_split_take)}\t Val: {len(valid_dataset.curr_split_take)}')
    logger.info(f"Learning rate: {cfg.TRAIN.LR} || Batch size: Train:{cfg.TRAIN.BATCH_SIZE}\t Val: {cfg.TEST.BATCH_SIZE}")

    ############ Train model & validation ###########
    best_val_loss = 1e2
    best_train_loss = 1e2
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        # train for one epoch
        logger.info(f'############# Starting Epoch {epoch} #############')
        train_loss = train(cfg, train_generator, model, criterion, optimizer, epoch, final_output_dir, device, logger, writer_dict)

        # evaluate on validation set
        val_loss = validate(cfg, val_generator, model, criterion, final_output_dir, device, logger, writer_dict)

        # Save best model weight
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model weight
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(final_output_dir, f"POTTER-HandPose-{cfg.DATASET.DATASET}-val.pt"))
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            # Save model weight
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, os.path.join(final_output_dir, f"POTTER-HandPose-{cfg.DATASET.DATASET}-train.pt"))


if __name__ == '__main__':
    args = parse_args_function()
    main(args)