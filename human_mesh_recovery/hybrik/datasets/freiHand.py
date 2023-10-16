import numpy as np
import os
from PIL import Image
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset

DATASET_MEANS = [0.3950, 0.4323, 0.2954]
DATASET_STDS = [0.1964, 0.1734, 0.1831]


def projectPoints(xyz, K):
    """
    Projects 3D coordinates into image space.
    Function taken from https://github.com/lmb-freiburg/freihand
    """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


class FreiHAND(Dataset):
    """
    Class to load FreiHAND dataset. Only training part is used here.
    Augmented images are not used, only raw - first 32,560 images

    Link to dataset:
    https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
    """

    def __init__(self, config, set_type="training", transform=None, chunk=1):
        """
        chunk (int): Number of chunks to be used from all freiHand dataset (4 chunks in total)
        """
        # Load in image info
        self.data_dir = config.DATASET.ROOT
        self.image_dir = os.path.join(self.data_dir, "training/rgb")
        self.all_img_dir = sorted(os.listdir(self.image_dir))
        # Load camera intrinsic matrix
        fn_K_matrix = os.path.join(self.data_dir, "training_K.json")
        with open(fn_K_matrix, "r") as f:
            self.K_matrix_all = np.array(json.load(f))
        # Load annotation
        fn_anno = os.path.join(self.data_dir, "training_xyz.json")
        with open(fn_anno, "r") as f:
            self.anno_all = np.array(json.load(f))
        self.chunk_length = len(self.anno_all)

        # Misc
        self.chunk = chunk
        self.image_size = np.array(config.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(config.MODEL.EXTRA.HEATMAP_SIZE)
        self.num_joints = config.MODEL.NUM_JOINTS
        self.target_type = config.MODEL.EXTRA.TARGET_TYPE
        self.sigma = config.MODEL.EXTRA.SIGMA

        # Image transform
        self.image_transform = transform

        # Train/val/test split
        split = [config.DATASET.TRAIN_RATIO, config.DATASET.VAL_RATIO, config.DATASET.TEST_RATIO]
        if set_type == "train":
            n_start = 0
            n_end = int(self.chunk_length*split[0])
        elif set_type == "val":
            n_start = int(self.chunk_length*split[0])
            n_end = int(self.chunk_length*split[0]) + int(self.chunk_length*split[1])
        elif set_type == 'test':
            n_start = int(self.chunk_length*split[0]) + int(self.chunk_length*split[1])
            n_end = self.chunk_length
        else:
            raise Exception(f"Invalid type: {set_type}. Must be within [train, val, test]")
        # Load in K and annotation for current split
        self.K_matrix = self.K_matrix_all[n_start:n_end]
        self.anno = self.anno_all[n_start:n_end]
        # Load in image names for current split based on number of chunks
        self.image_names = []
        for chunk_idx in range(self.chunk):
            offset = self.chunk_length*chunk_idx
            self.image_names.extend(self.all_img_dir[n_start+offset:n_end+offset])

        # Compute all 2D joints
        self.joints = []
        for i in range(len(self.anno)):
            self.joints.append(projectPoints(self.anno[i], self.K_matrix[i]))


    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):
        # Image (normalized)
        image_name = self.image_names[idx]
        input = np.array(Image.open(os.path.join(self.image_dir, image_name)))
        if self.image_transform:
            input = self.image_transform(input)

        # 2D joints array
        ann_idx = idx % len(self.anno)
        keypoints = self.joints[ann_idx]

        # Target hetmap
        target, _ = self.generate_target(keypoints)
        target = torch.from_numpy(target)

        return input, keypoints, target


    def generate_target(self, joints):
        '''
        :param joints:  [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)
            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
