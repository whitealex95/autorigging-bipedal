import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.rotation_util import compute_rotation_matrix_from_ortho6d, compute_geodesic_distance_from_two_matrices, compute_L2_distance_from_two_matrices
from utils.bvh_utils import compute_bindpose_from_bvh_animation, get_global_bvh_rotations_torch
from utils.joint_util import transform_rel2glob, toSE3
from utils.skin_util import vertex_transform, mesh_transform

blue = lambda x: '\033[94m' + x + '\033[0m'

from utils.loss_utils import JointsMSELoss, compute_mpjpe, compute_bone_symmetry_loss
from models.networks.bvh_simple import SimpleBVHNet, SimpleBVHNet_BN


class MixamoBVHModel(nn.Module):
    # Input: sdf Vox, Output: joint heatmap
    def __init__(self, configs):
        super(MixamoBVHModel, self).__init__()
        self.configs = configs
        self.writer = SummaryWriter
        self.losses = defaultdict(torch.FloatTensor)
        if configs['network'] == 'simple':
            self.network = SimpleBVHNet(in_dim=22*3, out_dim=22*6, hidden_dim=[512, 256, 256])
        elif configs['network'] == 'simple_bn':
            self.network = SimpleBVHNet_BN(in_dim=22*3, out_dim=22*6, hidden_dim=[512, 256, 256])
        else:
            raise NotImplementedError

    def preprocess_position(self, input_position):
        # zero root position
        if 'zero_root' in self.configs.keys() and self.configs['zero_root'] is True:
            input_position = input_position - input_position[:, :1, :]
        # should we normalize?
        return input_position

    def postprocess_rotation(self, pred_rotation):
        if self.configs['rel_rot']: # network is outputting relative joint rotation
            # pred_rotation1 = transform_rel2glob(pred_rotation)
            pred_rotation = get_global_bvh_rotations_torch(pred_rotation)

        return pred_rotation


    def forward(self, input_position):
        ######## Becareful to uncomment this when zeroing input  ########
        input_position = self.preprocess_position(input_position)
        ######## Becareful to uncomment this when zeroing input  ########
        pred_rotation_param = self.network(input_position)  # [B, J, nParam] -> [B, J, rotParam]
        pred_rotation = self.to_rotation_matrix(pred_rotation_param)
        pred_rotation = self.postprocess_rotation(pred_rotation)
        return pred_rotation

    def to_rotation_matrix(self, rotation_param):
        # Todo: more representations
        # if True or self.configs['rotation_representation'] == '6D':
        return compute_rotation_matrix_from_ortho6d(rotation_param)
        # else:
        #     raise NotImplementedError


    def compute_loss(self, pred_rotation, target_rotation, average=True):
        # SO3, SO3 -> radians
        theta = self.theta = compute_geodesic_distance_from_two_matrices(pred_rotation, target_rotation)
        if average:
            return torch.mean(theta)
        return theta


    def compute_accuracy(self, input_position, pred_rotation, target_rotation, average=True):
        target_bindpose_position = compute_bindpose_from_bvh_animation(input_position, target_rotation)
        pred_bindpose_position = compute_bindpose_from_bvh_animation(input_position, pred_rotation)
        bindpose_mpjpe = compute_mpjpe(pred_bindpose_position, target_bindpose_position)  # mean per joint position error [B,]
        if average:
            return torch.mean(bindpose_mpjpe)
        return bindpose_mpjpe

    def print_loss(self, loss, acc, epoch, i, num_batch):
        mode = 'Train' if self.training else 'Eval'
        status, end = (blue(mode) + ' {}'.format(epoch), '') if not self.training else (mode + ' {}: {}/{}'.format(epoch, i, num_batch), '')
        print("\r[" + status + "] ",
              "loss : {:.8f}, acc : {:.4f}".format(loss, acc), end=end)

    def write_summary(self, losses_dict, step=None, epoch=None, writer=None):
        mode = 'Train' if self.training else 'Eval'
        if step is None and epoch is not None:
            mode += '/epoch'
            step=epoch
        l = defaultdict(float)
        for key, val in losses_dict.items():
            if isinstance(val, list):
                l[key] = val[-1]
            else:
                l[key] = val
            writer.add_scalar(f'{mode}/{key}', l[key], step)
