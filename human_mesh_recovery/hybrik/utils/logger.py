import torch
import os
import logging
import time
from pathlib import Path


def board_writing(writer, loss, acc, iterations, dataset='Train'):
    writer.add_scalar(
        '{}/Loss'.format(dataset), loss, iterations)
    writer.add_scalar(
        '{}/acc'.format(dataset), acc, iterations)


def debug_writing(writer, outputs, labels, inputs, iterations):
    tmp_tar = torch.unsqueeze(labels.cpu().data[0], dim=1)
    # tmp_out = torch.unsqueeze(outputs.cpu().data[0], dim=1)

    tmp_inp = inputs.cpu().data[0]
    tmp_inp[0] += 0.406
    tmp_inp[1] += 0.457
    tmp_inp[2] += 0.480

    # tmp_inp[0] += torch.sum(F.interpolate(tmp_tar, scale_factor=4, mode='bilinear', align_corners=True), dim=0)[0]
    tmp_inp.clamp_(0, 1)
    writer.add_image('Data/input', tmp_inp, iterations)
    for i in range(18, 18 + 17):
        writer.add_image('Data/target', tmp_tar[i], iterations)


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)