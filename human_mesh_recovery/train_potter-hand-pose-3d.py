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


def train(config, train_loader, model, criterion, optimizer, epoch, output_dir, device, logger, writer_dict):
    loss_2d = AverageMeter()
    loss_3d = AverageMeter()
    total_loss = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    train_loader = tqdm(train_loader, dynamic_ncols=True)
    print_interval = len(train_loader) // config.TRAIN_PRINT_NUM

    for i, (input, pose_2d_gt, hm_gt, target_weight, pose_3d_gt, vis_flag, _) in enumerate(train_loader):
        # compute output
        input = input.to(device)
        hm_pred, pose_3d_pred = model(input)
        hm_gt = hm_gt.to(device)
        # Assign None kpts as zero
        pose_3d_gt[~vis_flag] = 0
        pose_3d_gt = pose_3d_gt.to(device)
        vis_flag = vis_flag.to(device)

        hm_loss, pose_3d_loss = criterion(hm_pred, hm_gt, pose_3d_pred, pose_3d_gt, vis_flag)
        loss = pose_3d_loss #hm_loss + pose_3d_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        # loss_2d.update(hm_loss.item())
        loss_3d.update(pose_3d_loss.item())
        total_loss.update(loss.item())

        # # Caculate 2D PCK as accuracy
        # _, avg_acc, cnt, pred = accuracy(hm_pred.detach().cpu().numpy(),
        #                                  hm_gt.detach().cpu().numpy())
        # acc.update(avg_acc, cnt)

        # Log info
        if (i+1) % print_interval == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  '2D Loss {loss_2d.val:.5f} ({loss_2d.avg:.5f})\t' \
                  '3D Loss {loss_3d.val:.5f} ({loss_3d.avg:.5f})\t' \
                  'Total loss {total_loss.val:.5f} ({total_loss.avg:.5f})\t' \
                  '2D PCK Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i+1, len(train_loader),
                      loss_2d=loss_2d, loss_3d=loss_3d, total_loss=total_loss, acc=acc)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('Loss/train', total_loss.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

            # # Save debug images
            # prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i+1)
            # debug_pose_2d_gt = pose_2d_gt.clone()
            # debug_pose_2d_gt[~vis_flag] = 0
            # meta = {'joints': debug_pose_2d_gt, 
            #         "joints_vis": target_weight}
            # save_debug_images(config, input.cpu(), meta, hm_gt.cpu(), pred*4, hm_pred.cpu(), prefix)
    return total_loss.avg


def validate(config, val_loader, model, criterion, output_dir, device, logger, writer_dict):
    loss_2d = AverageMeter()
    loss_3d = AverageMeter()
    total_loss = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        val_loader = tqdm(val_loader, dynamic_ncols=True)
        for i, (input, pose_2d_gt, hm_gt, target_weight, pose_3d_gt, vis_flag, _) in enumerate(val_loader):
            # compute output
            input = input.to(device)
            hm_pred, pose_3d_pred = model(input)
            hm_gt = hm_gt.to(device)
            # Assign None kpts as zero
            pose_3d_gt[~vis_flag] = 0
            pose_3d_gt = pose_3d_gt.to(device)
            vis_flag = vis_flag.to(device)

            hm_loss, pose_3d_loss = criterion(hm_pred, hm_gt, pose_3d_pred, pose_3d_gt, vis_flag)
            loss = pose_3d_loss #hm_loss + pose_3d_loss

            # measure accuracy and record loss
            # loss_2d.update(hm_loss.item())
            loss_3d.update(pose_3d_loss.item())
            total_loss.update(loss.item())

            # # Caculate 2D PCK as accuracy
            # _, avg_acc, cnt, pred = accuracy(hm_pred.detach().cpu().numpy(),
            #                                 hm_gt.detach().cpu().numpy())
            # acc.update(avg_acc, cnt)

        # Log info
        msg = 'Test: [{0}/{1}]\t' \
                '2D Loss {loss_2d.val:.5f} ({loss_2d.avg:.5f})\t' \
                '3D Loss {loss_3d.val:.5f} ({loss_3d.avg:.5f})\t' \
                'Total loss {total_loss.val:.5f} ({total_loss.avg:.5f})\t' \
                '2D PCK Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i+1, len(val_loader),
                    loss_2d=loss_2d, loss_3d=loss_3d, total_loss=total_loss, acc=acc)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('Loss/val', total_loss.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1
                
                # # Save debug images
                # prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i+1)
                # debug_pose_2d_gt = pose_2d_gt.clone()
                # debug_pose_2d_gt[~vis_flag] = 0
                # meta = {'joints': debug_pose_2d_gt, 
                #         "joints_vis": target_weight}
                # save_debug_images(config, input.cpu(), meta, hm_gt.cpu(), pred*4, hm_pred.cpu(), prefix)
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
        load_pretrained_weights(model, torch.load(pretrained_hand_pose_CKPT))
        logger.info('Loaded pretrained hand pose estimation weight')
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
                                 split='val', 
                                 transform=transform,
                                 use_preset=args.use_preset)
    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    logger.info(f'Number of takes: Train: {len(train_dataset.curr_split_take)}\t Val: {len(valid_dataset.curr_split_take)}')
    logger.info(f"Learning rate: {cfg.TRAIN.LR} || Batch size: Train:{cfg.TRAIN.BATCH_SIZE}\t Val: {cfg.TEST.BATCH_SIZE}")

    ############ Train model & validation ###########
    best_val_loss = 1e2
    best_train_loss = 1e2
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        # train for one epoch
        logger.info(f'############# Starting Epoch {epoch} #############')
        train_loss = train(cfg, train_loader, model, criterion, optimizer, epoch, final_output_dir, device, logger, writer_dict)

        # evaluate on validation set
        val_loss = validate(cfg, valid_loader, model, criterion, final_output_dir, device, logger, writer_dict)

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