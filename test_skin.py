import os
import argparse
import yaml

import torch.utils.data
import numpy as np
from torch_geometric.data import DataLoader

from datasets.mixamo_skin_dataset import MixamoSkinDataset
from models.mixamo_skin_model import MixamoMeshSkinModel

parser = argparse.ArgumentParser(description="need trained_model, its config, joint_dir")
parser.add_argument('--model', required=True)
parser.add_argument('--config', required=True)
parser.add_argument('--vol_geo_dir', required=True)
args = parser.parse_args()

# configs = get_configs_from_arguments(testing=True)
with open(args.config) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
configs['model'] = args.model
configs['vol_geo_dir'] = args.vol_geo_dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dataset = MixamoSkinDataset(data_dir=configs['data_dir'], split='test_models', vol_geo_dir=configs['vol_geo_dir'], preprocess=configs['preprocess'], datatype=configs['datatype'], configs=configs, test_all=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=int(configs['workers']), pin_memory=True)

model = MixamoMeshSkinModel(configs, num_joints=configs['num_joints'], use_bn=configs['use_bn'])
model.to(device)

checkpoint = torch.load(configs['model'])
model.load_state_dict(checkpoint['model_state_dict'])
print("[Info] Loaded model parameters from " + configs['model'])

with torch.no_grad():
    print("Testing")
    for data in test_dataloader:
        data = [dat.to(device) for dat in data[:-2]] + list(data[-2:])
        mesh, gt_jm, gt_ibm, character_name, motion_name = data
        pred_skin_logit = model(mesh)

        batch_size = mesh.batch.max().item() + 1
        for i in range(batch_size):
            pred_skin = torch.exp(pred_skin_logit)[mesh.batch==i].cpu().detach().numpy()
            character_name, motion_name = character_name[i], motion_name[i]
            print(character_name, motion_name)

            if not os.path.exists(os.path.join(configs['log_dir'], 'test', character_name)):
                os.makedirs(os.path.join(configs['log_dir'], 'test', character_name))

            np.savetxt(os.path.join(configs['log_dir'], 'test', character_name, motion_name + '_skin.csv'),
                        pred_skin, delimiter=',')
