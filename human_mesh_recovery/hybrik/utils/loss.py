import torch.nn as nn
import torch
import numpy as np


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
    

class Pose3DLoss(nn.Module):
    def __init__(self):
        super(Pose3DLoss, self).__init__()

    def forward(self, hm_pred, hm_gt, pose_3d_pred, pose_3d_gt, vis_flag):
        # Flatten heatmap along spatial dimension
        assert hm_pred.shape == hm_gt.shape and len(hm_pred.shape) == 4 # (N, K, H, W)
        batch_size, num_joints = hm_pred.shape[:2]
        heatmaps_pred = hm_pred.reshape(batch_size, num_joints, -1)
        heatmaps_gt = hm_gt.reshape(batch_size, num_joints, -1)

        # Compute MSE loss between pred and gt 2D heatmap for only visible kpts
        hm_diff = heatmaps_pred - heatmaps_gt
        hm_loss = torch.mean(hm_diff**2, axis=2) * vis_flag
        hm_loss = torch.sum(hm_loss) / torch.sum(vis_flag)

        # Compute MSE loss between pred and gt 3D hand joints for only visible kpts
        assert pose_3d_pred.shape == pose_3d_gt.shape and len(pose_3d_pred.shape) == 3 # (N, K, dim)
        pose_3d_diff = pose_3d_pred - pose_3d_gt
        pose_3d_loss = torch.mean(pose_3d_diff**2, axis=2) * vis_flag
        pose_3d_loss = torch.sum(pose_3d_loss) / torch.sum(vis_flag)

        return hm_loss, pose_3d_loss
    

def mpjpe(predicted, target, num=None):
    """
    Mean per-joint position error (i.e. mean Euclidean distance) from 
    https://github.com/zhaoweixi/GraFormer/blob/main/common/loss.py.
    Modified s.t. it could compute MPJPE for only those valid keypoints (where 
    # of visible keypoints = num)
    """
    assert predicted.shape == target.shape
    pjpe = torch.norm(predicted - target, dim=len(target.shape) - 1)
    mpjpe = torch.sum(pjpe) / num if num else torch.mean(pjpe)
    return mpjpe


def p_mpjpe(predicted, target, num=None):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    # Convert to Numpy because this metric needs numpy array
    predicted = predicted.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    pjpe = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
    p_mpjpe = np.sum(pjpe) / num if num else np.mean(pjpe)
    return p_mpjpe