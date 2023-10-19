import argparse

def parse_args_function():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--cfg_file",
        default='configs/ego4d/potter_pose_3d_ego4d.yaml',
        help="Config file path"
    )

    parser.add_argument(
        "--pretrained_ckpt",
        default=None,
        help="Pretrained potter-hand-pose-3d checkpoint"
    )

    parser.add_argument(
        "--cls_ckpt",
        default='eval/cls_s12.pth',
        help="Pretrained potter-cls checkpoint path"
    )

    parser.add_argument(
        "--anno_type",
        default='annotation',
        help="Use manual labelled or automatically generated GT data"
    )

    parser.add_argument(
        "--use_preset",
        action='store_true',
        help="Whether use pre-selected or offical split takes"
    )

    parser.add_argument(
        "--gpu_number",
        type=int,
        nargs='+',
        default = [0],
        help="Identifies the GPU number to use."
    )


    args = parser.parse_args()
    return args