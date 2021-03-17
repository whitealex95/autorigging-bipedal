import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

blue = lambda x: '\033[94m' + x + '\033[0m'

from utils.loss_utils import softmax3d, normalize3d, compute_mpjpe, compute_bone_symmetry_loss
from utils.voxel_utils import get_final_preds, get_max_preds, get_final_preds_torch, get_max_preds_torch, downsample_heatmap
from models.networks.vox_hourglass import V2V_HG

class MixamoVoxModel(nn.Module):
    # Input: sdf Vox, Output: joint heatmap
    def __init__(self, configs):
        super(MixamoVoxModel, self).__init__()
        self.configs = configs
        self.writer = SummaryWriter
        self.losses = defaultdict(torch.FloatTensor)
        self.network = V2V_HG(input_channels=1, feature_channels=configs['feature_channels'], n_stack=configs['n_stack'], n_joint=22,
                                downsample=self.configs['downsample'], configs=configs)

        if self.configs['activation'] == 'sigmoid':
            self.activation = F.sigmoid
        elif self.configs['activation'] == 'softmax':
            self.activation = softmax3d
        elif self.configs['activation'] == 'none':
            self.activation = None
        else:
            __import__('pdb').set_trace()
            print('Unrecognized activation')
            self.activation = None

    def forward(self, vox):
        heatmap = self.network(vox)
        if self.activation is not None:
            heatmap = self.activation(heatmap)
        return heatmap

    def compute_loss(self, pred_heatmap, target_heatmap, mask=None):
        loss_type = self.configs['loss_type']
        if self.configs['normalize_heatmap']:
            target_heatmap = normalize3d(target_heatmap)
        if self.configs['downsample'] > 1:
            target_heatmap = downsample_heatmap(target_heatmap, self.configs['downsample'])
        if loss_type == 'ce':
            assert self.configs['activation'] not in ['sigmoid', 'softmax']
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_heatmap, target_heatmap)
            # loss = torch.nn.BCEWithLogitsLoss()(pred_heatmap, target_heatmap)
        elif loss_type == 'ce_mask':
            assert self.configs['activation'] not in ['sigmoid', 'softmax']
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_heatmap, target_heatmap,
                weight=mask, reduction='sum') / mask.sum() / 22  # divide by n_joint
        elif loss_type == 'mse':
            loss = torch.mean((pred_heatmap - target_heatmap) ** 2)
        else:
            print("Unrecognized loss type")
            raise NotImplementedError

        if self.configs['loss_symmetry'] != 0:
            loss += self.configs['loss_symmetry'] * compute_bone_symmetry_loss(pred_heatmap)
        return loss


    def compute_accuracy(self, pred_heatmap, target_heatmap, JM4x4_rel, meta, average=True):
        """
        Input: torch
        output: np
        """
        # JM4x4_global = transform_rel2glob(JM4x4_rel)
        # target_coords = JM4x4_global[:, :, :3, 3]
        device = pred_heatmap.device
        translate, scale = torch.stack(meta['translate'], axis=1).to(device)[:, None, :], \
                           meta['scale'].to(device)[:, None, None]
        target_coords, _ = get_final_preds_torch(target_heatmap, translate, scale)
        pred_coords, _ = get_final_preds_torch(pred_heatmap, translate, scale * self.configs['downsample'])
        # if not is_mean:
        #     a = (pred_coords - target_coords) * (pred_coords - target_coords)
        #     return torch.einsum('ijk->i', a)/float(torch.prod(torch.Tensor(a.shape[1:])))
        mpjpe = compute_mpjpe(pred_coords, target_coords)  # mean per joint position error [B,]
        if average:
            return torch.mean(mpjpe)
        else:
            return mpjpe


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
