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
import matplotlib.pyplot as plt
from torchsummary import summary
from hybrik.models.PoolAttnHR_Pose import PoolAttnHR_Pose
from hybrik.datasets.coco import *
from hybrik.datasets.freiHand import FreiHAND
from torch.utils.data import Dataset, DataLoader
from hybrik.utils.loss import JointsMSELoss
from hybrik.utils.functions import AverageMeter
from hybrik.utils.vis import save_debug_images
from hybrik.utils.evaluate import accuracy
from hybrik.datasets.cocoHand import COCOHandDataset


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('backbone.'):
            k = k[9:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print(f'Successfully loaded {len(matched_layers)} pretrained parameters')


def train(config, train_loader, model, criterion, optimizer, epoch, output_dir, device, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    train_loader = tqdm(train_loader, dynamic_ncols=True)
    for i, (input, gt_kpts, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        input = input.to(device)
        output = model(input)
        target = target.to(device)
        target_weight = None

        loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log info
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            # Save debug images
            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            meta = {'joints':gt_kpts, 
                    "joints_vis":torch.ones(gt_kpts.shape[0], gt_kpts.shape[1], 1)}
            save_debug_images(config, input.cpu(), meta, target.cpu(), pred*4, output.cpu(),
                              prefix)


def validate(config, val_loader, model, criterion, output_dir, device, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        val_loader = tqdm(val_loader, dynamic_ncols=True)
        for i, (input, gt_kpts, target) in enumerate(val_loader):
            # compute output
            input = input.to(device)
            output = model(input)

            target = target.to(device)
            target_weight = None

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                meta = {'joints':gt_kpts,
                        "joints_vis":torch.ones(gt_kpts.shape[0], gt_kpts.shape[1], 1)}
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix) 
    return losses.avg
    


def main():
    torch.cuda.empty_cache()
    cfg_file = '/home/jinxu/code/POTTER/human_mesh_recovery/configs/freihand/potter_pose_freihand.yaml'
    pretrained_hand_pose_CKPT = 'output/freihand/PoolAttnHRCam_Pose/potter_pose_freihand_chunk=1/POTTER-HandPose-freiHand.pt'
    CLS_CKPT = '/home/jinxu/code/POTTER/human_mesh_recovery/eval/cls_s12.pth'
    cfg = update_config(cfg_file)
    gpu_index = 2
    device = torch.device(f"cuda:{gpu_index}")
    logger, final_output_dir, _ = create_logger(cfg, cfg_file, 'train')

    ############ MODEL ###########
    # Define model
    model = PoolAttnHR_Pose(**cfg.MODEL)
    # Load pretrained cls_weight or available hand pose weight
    cls_weight = torch.load(CLS_CKPT)
    if pretrained_hand_pose_CKPT:
        load_pretrained_weights(model, torch.load(pretrained_hand_pose_CKPT))
        print('Loaded pretrained hand pose estimation weight')
    else:
        load_pretrained_weights(model.poolattnformer_pose.poolattn_cls, cls_weight)
        print('Load POTTER-cls weight')
    model = model.to(device)

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
    # Load dataset
    if cfg.DATASET.DATASET == 'coco':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = COCOHandDataset(
            cfg,
            cfg.DATASET.ROOT,
            cfg.DATASET.TRAIN_SET,
            False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
        valid_dataset = COCOHandDataset(
            cfg,
            cfg.DATASET.ROOT,
            cfg.DATASET.TEST_SET,
            False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    elif cfg.DATASET.DATASET == 'freihand':
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3950, 0.4323, 0.2954],
                                std=[0.1964, 0.1734, 0.1831])
        ])
        train_dataset = FreiHAND(config=cfg, 
                                 set_type="train", 
                                 transform=data_transform,
                                 chunk=cfg.DATASET.CHUNK)
        valid_dataset = FreiHAND(config=cfg, 
                                 set_type="val", 
                                 transform=data_transform,
                                 chunk=cfg.DATASET.CHUNK)
    else:
        raise Exception(f"Not implemented {cfg.DATASET.DATASET} yet.")
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

    ############ Train model & validation ###########
    best_val_loss = 1e4
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        # train for one epoch
        logger.info(f'############# Starting Epoch {epoch} #############')
        train(cfg, train_loader, model, criterion, optimizer, epoch, final_output_dir, device, logger)

        # evaluate on validation set
        val_loss = validate(cfg, valid_loader, model, criterion, final_output_dir, device, logger)

        # Save best model weight
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model weight
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(final_output_dir, "POTTER-HandPose-freiHand.pt"))


if __name__ == '__main__':
    main()
