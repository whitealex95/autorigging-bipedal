import os
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import defaultdict

from utils.joint_util import save_jm, save_jm2
from utils.train_vox_utils import get_vox_dataloaders, train_vox, eval_vox, vis_vox, get_configs_from_arguments
from models.mixamo_vox_model import MixamoVoxModel
from time import time
from datasets.mixamo_vox_dataset import MixamoVoxDataset
from torch.utils.data import DataLoader

from utils.voxel_utils import get_final_preds, extract_joint_pos_from_heatmap_softargmax, extract_joint_pos_from_heatmap, downsample_single_heatmap
from utils.joint_util import transform_rel2glob
from utils.loss_utils import normalize3d

configs = get_configs_from_arguments(testing=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dataset = MixamoVoxDataset('test_models.txt', configs, reduce_motion=False, use_front_faced=True)  # front faced only for test
test_dataloader = DataLoader(test_dataset, batch_size=configs['batch_size'], shuffle=False, num_workers=int(configs['workers']), pin_memory=True)

model = MixamoVoxModel(configs)
model.to(device)

checkpoint = torch.load(configs['model'])
model.load_state_dict(checkpoint['model_state_dict'])
print("[Info] Loaded model parameters from " + configs['model'])


model.eval()
with torch.no_grad():
    for data in test_dataloader:
        data = [dat.to(device).float() for dat in data[:-1]] + [data[-1]]
        bin_vox_padded, sdf_vox_padded, target_heatmap, JM4x4, meta = data
        if configs['normalize_heatmap']:
            target_heatmap = normalize3d(target_heatmap)
        pred_heatmap_logit = model(sdf_vox_padded)
        if configs['loss_type'].endswith('mask'):
            pred_heatmap = torch.sigmoid(pred_heatmap_logit) * bin_vox_padded
        else:
            pred_heatmap = torch.sigmoid(pred_heatmap_logit)
        acc = joint_acc = model.compute_accuracy(pred_heatmap, target_heatmap, JM4x4, meta, average=False)

        character_name, motion_name = meta['character_name'], meta['motion_name']
        batch_size = bin_vox_padded.shape[0]
        pred_coords, _ = get_final_preds(pred_heatmap, meta['translate'], meta['scale']* configs['downsample'])
        target_heatmap_coords, _ = get_final_preds(target_heatmap, meta['translate'], meta['scale'])
        JM4x4_global = transform_rel2glob(JM4x4)
        target_coords = JM4x4_global[:, :, :3, 3]

        for i in range(batch_size):
            character_name_i, motion_name_i = character_name[i], motion_name[i]
            print(character_name_i, motion_name_i)
            pred_heatmap_i = pred_heatmap[i].cpu().numpy()
            target_heatmap_i = target_heatmap[i].cpu().numpy()
            pred_coords_i = pred_coords[i]
            target_coords_i = target_coords[i].cpu().numpy()
            target_heatmap_coords_i = target_heatmap_coords[i]
            acc_i = acc[i].cpu().numpy()
            if not os.path.exists(os.path.join(configs['log_dir'], 'test', character_name_i)):
                os.makedirs(os.path.join(configs['log_dir'], 'test', character_name_i))

            # np.save(os.path.join(configs['log_dir'], 'test', character_name_i, motion_name_i + '_pred_hm'),pred_heatmap_i)
            # np.save(os.path.join(configs['log_dir'], 'test', character_name_i, motion_name_i + '_gt_hm'),target_heatmap_i)
            np.save(os.path.join(configs['log_dir'], 'test', character_name_i, motion_name_i + '_info'),{
                'pred_coords': pred_coords_i,
                'target_coords': target_coords_i,
                'target_heatmap_coords': target_heatmap_coords_i,
                'acc': acc_i
            })

            scale_i, translate_i = meta['scale'][i].cpu().numpy(), torch.stack(meta['translate']).T[i].cpu().numpy()
            center_trans_i = meta['center_trans'][i].cpu().numpy()
            use_downsample = False
            mask = None
            if configs['downsample'] > 1:
                use_downsample = True
                mask = downsample_single_heatmap(torch.Tensor(bin_vox_padded[i][None,...].cpu()), 2).numpy()
            pred_heatmap_joint_pos = extract_joint_pos_from_heatmap(pred_heatmap_i, scale_i, translate_i, use_downsample=use_downsample, center_trans=center_trans_i, mask=None)
            pred_heatmap_joint_pos_soft = extract_joint_pos_from_heatmap_softargmax(pred_heatmap_i, scale_i, translate_i, use_downsample=use_downsample, center_trans=center_trans_i, mask=None)
            pred_heatmap_joint_pos_mask = extract_joint_pos_from_heatmap(pred_heatmap_i, scale_i, translate_i, use_downsample=use_downsample, center_trans=center_trans_i, mask=mask)

            np.save(os.path.join(configs['log_dir'], 'test', character_name_i, motion_name_i + '_joint'),{
                'pred_coords': pred_coords_i,
                'pred_heatmap_joint_pos': pred_heatmap_joint_pos,
                'pred_heatmap_joint_pos_soft': pred_heatmap_joint_pos_soft,
                'pred_heatmap_joint_pos_mask': pred_heatmap_joint_pos_mask
            })
            np.save(os.path.join(configs['log_dir'], 'test', character_name_i, motion_name_i + '_joint_pos_mask'), pred_heatmap_joint_pos_mask)


