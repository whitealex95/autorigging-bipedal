import os
import numpy as np

import yaml
import argparse

import torch.utils.data
from torch.utils.data import DataLoader

from models.mixamo_bvh_model import MixamoBVHModel
from datasets.mixamo_bvh_dataset import MixamoBVHDataset

parser = argparse.ArgumentParser(description="need trained_model, its config, joint_dir")
parser.add_argument('--model', required=True)
parser.add_argument('--config', required=True)
parser.add_argument('--joint_path', required=True)
args = parser.parse_args()

# configs = get_configs_from_arguments(testing=True)
with open(args.config) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
configs['model'] = args.model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

joint_path = 'logs/vox/vox_ori_all_s4.5_HG_mean_stack2_down2_lr3e-4_b4_ce/test'

test_dataset = MixamoBVHDataset('test_models.txt', configs, use_front_faced=True, joint_path=joint_path)  # front faced only for test
test_dataloader = DataLoader(test_dataset, batch_size=configs['batch_size'], shuffle=False, num_workers=int(configs['workers']), pin_memory=True)

model = MixamoBVHModel(configs)
model.to(device)

checkpoint = torch.load(configs['model'])
model.load_state_dict(checkpoint['model_state_dict'])
print("[Info] Loaded model parameters from " + configs['model'])

model.eval()
with torch.no_grad():
    for data in test_dataloader:
        data = [dat.to(device).float() for dat in data[:-1]] + [data[-1]]
        input_position, target_rotation, meta = data  # forget target_rotation....
        for key in meta:
            if isinstance(meta[key], torch.Tensor):
                meta[key].to(device).float()
        pred_rotation = model(input_position)

        character_name, motion_name = meta['character_name'], meta['motion_name']
        batch_size = input_position.shape[0]
        for i in range(batch_size):
            character_name_i, motion_name_i = character_name[i], motion_name[i]
            if not os.path.exists(os.path.join(configs['log_dir'],'test'+'_'+configs['model'].split('/')[-1].split('_')[-1].split('.pth')[0], character_name_i)):
                os.makedirs(os.path.join(configs['log_dir'], 'test'+'_'+configs['model'].split('/')[-1].split('_')[-1].split('.pth')[0], character_name_i))

            model_epoch = configs['model'].split('/')[-1].split('_')[-1].split('.pth')[0]
            np.save(os.path.join(configs['log_dir'], 'test'+'_'+ model_epoch, character_name_i, motion_name_i + '_pred_rot'),pred_rotation[i].cpu().detach())
            np.save(os.path.join(configs['log_dir'], 'test'+'_'+ model_epoch, character_name_i, motion_name_i + '_target_rot'),target_rotation[i].cpu().detach())

print("[Info] Inference Done")
