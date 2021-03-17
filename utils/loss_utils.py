import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.voxel_utils import get_softmax_preds
from utils.joint_util import joint_pos2bone_len

def normalize3d(batch):
    """
    batch (B, J, N, N, N)
    """
    B, J = batch.shape[0], batch.shape[1]

    norm = batch.reshape(B, J, -1).sum(dim=-1, keepdim=True)
    return batch.reshape(B, J, -1).div(norm).reshape(batch.shape)
    

def softmax3d(batch):
    batch_size = batch.shape[0]
    joint_num = batch.shape[1]
    return F.softmax(batch.reshape(batch_size, joint_num, -1), dim=2).reshape(batch.shape)
    # F.softmax(batch.view(batch_size, joint_num, -1)).view(batch.shape)

def compute_mpjpe(pred_coords, target_coords):
    """
    batch (B, J, 3)
    """
    if isinstance(pred_coords, np.ndarray):
        return np.mean(np.sqrt(np.mean((pred_coords - target_coords) ** 2, axis=-1)))
    else:
        return torch.mean(torch.sqrt(torch.sum((pred_coords - target_coords) ** 2, dim=-1)), dim=-1)

def compute_joint_ce_loss(joint_heatmap, target_heatmap):
    """
    (B, J, 88, 88, 88)
    """
    B, J = joint_heatmap.shape[0], joint_heatmap.shape[1]
    loss_joint = F.binary_cross_entropy_with_logits(joint_heatmap, target_heatmap)
    return loss_joint

class JointsMSELoss(nn.Module):
    # Not used
    def __init__(self):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)

    def forward(self, pred, target):
        batch_size = pred.size(0)
        num_joints = pred.size(1)
        heatmaps_pred = pred.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

def compute_bone_symmetry_loss(joint_heatmap):
    B, J, H, W, D = joint_heatmap.shape
    device = joint_heatmap.device
    joint_pos = get_softmax_preds(joint_heatmap)
    assert J == 22
    bone_len = joint_pos2bone_len(joint_pos)
    """
    Symmetry:
    0-1
    6-7
    9,10,11 - 12, 13, 14
    15,16,17 - 18,19,20
    """
    left_symmetry_idx = [0, 6, 9, 10, 11, 15, 16, 17]
    right_symmetry_idx = [1, 7, 12, 13, 14, 18, 19, 20]
    # MSE Loss
    loss = 0
    for i in range(len(left_symmetry_idx)):
        loss += (bone_len[...,left_symmetry_idx[i]] - bone_len[...,right_symmetry_idx[i]])**2
    return torch.mean(loss)


