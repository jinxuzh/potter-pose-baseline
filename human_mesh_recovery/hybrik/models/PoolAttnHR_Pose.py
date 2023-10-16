from collections import namedtuple
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from .pool.poolattnformer_HR import PoolAttnFormer_hr, load_pretrained_weights, GroupNorm

ModelOutput = namedtuple(
    typename='ModelOutput',
    field_names=['pred_shape', 'pred_theta_mats', 'pred_phi', 'pred_delta_shape', 'pred_leaf',
                 'pred_uvd_jts', 'pred_xyz_jts_29', 'pred_xyz_jts_24', 'pred_xyz_jts_24_struct',
                 'pred_xyz_jts_17', 'pred_vertices', 'maxvals', 'cam_scale', 'cam_trans', 'cam_root',
                 'uvd_heatmap', 'transl', 'img_feat', 'all_HR_stage']
)
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)


def norm_heatmap(norm_type, heatmap):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError



class PoolAttnHR_Pose(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(PoolAttnHR_Pose, self).__init__()
        self.deconv_dim = kwargs['NUM_DECONV_FILTERS']
        # self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.img_H = kwargs['IMAGE_SIZE'][0]
        self.img_W = kwargs['IMAGE_SIZE'][1]

        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.layers = kwargs['EXTRA']['LAYERS']
        self.embed_dims = kwargs['EXTRA']['EMBED_DIMS']
        self.mlp_ratios = kwargs['EXTRA']['MLP_RATIOS']
        self.drop_rate = kwargs['EXTRA']['DROP_RATE']
        self.drop_path_rate = kwargs['EXTRA']['DROP_PATH_RATE']
        self.pretrained = kwargs['EXTRA']['PRETRAINED']

        self.height_dim = kwargs['EXTRA']['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['EXTRA']['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32

        ### backbone:   POTTER block pretrained with image size of [256,256]
        ###  Due to the limited time and resources, I didn't retrain the entire network
        ###  with the correct names of each module shows in the paper
        ### Sorry for the inconvenience
        img_size = [self.img_H, self.img_W]
        self.poolattnformer_pose = PoolAttnFormer_hr(img_size, layers=self.layers, embed_dims=self.embed_dims,
                                               mlp_ratios=self.mlp_ratios, drop_rate=self.drop_rate,
                                               drop_path_rate=self.drop_path_rate,
                                               use_layer_scale=True, layer_scale_init_value=1e-5,)

        if self.pretrained != "None":
            pt_checkpoint = torch.load(self.pretrained, map_location=lambda storage, loc: storage)
            # pt_checkpoint = pt_checkpoint["model"]
            # model.load_state_dict(pt_checkpoint, False)
            self.poolattnformer_pose = load_pretrained_weights(self.poolattnformer_pose, pt_checkpoint)

        self.norm1 = GroupNorm(256)


        ################ pose regression
        self.up_sample = nn.Sequential(
            nn.Conv2d(self.embed_dims[0], 256, 1),
            nn.GELU(),
        )

        self.final_layer = nn.Conv2d(
            256, self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)

        self.pose_layer =nn.Sequential(
            nn.Conv3d(
                self.num_joints, self.num_joints, 1),
            nn.GELU(),
            nn.GroupNorm(self.num_joints, self.num_joints),
            nn.Conv3d(
                self.num_joints, self.num_joints, 1),
        )

        self.norm2 = nn.BatchNorm2d(256) #GroupNorm(self.num_joints)


    def _initialize(self):
        for m in self.up_sample.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, flip_item=None, flip_output=False, **kwargs):
        batch_size = x.shape[0]
        # Forward through all branches
        x_feature, _, _ = self.poolattnformer_pose(x)  ######### ######### x0 torch.Size([B, 64, 64, 64]  /

        # predict pose
        out = self.up_sample(x_feature)  ######### out1 torch.Size([B, 256, 64, 64])
        out = self.norm1(out)
        out = self.final_layer(out)  ######### out2 torch.Size([B, num_joints*emb_dim, 64, 64])
        out = self.pose_layer(out.reshape(out.shape[0], self.num_joints, self.depth_dim, out.shape[2], out.shape[3]))
        out = out.reshape(out.shape[0], self.num_joints, out.shape[2], out.shape[3], -1) # (N, num_joints, 64, 64, 64)
        out = out.sum(2) # (N, num_joints, 64, 64) Elementwisely added along channel direction
        return out

    def forward_gt_theta(self, gt_theta, gt_beta):

        output = self.smpl(
            pose_axis_angle=gt_theta,
            betas=gt_beta,
            global_orient=None,
            return_verts=True
        )

        return output
