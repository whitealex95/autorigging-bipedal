import os
import json
import sys

import numpy as np
from tqdm import tqdm


import torch
import torch.utils.data as data

sys.path.insert(0, '..')
sys.path.insert(0, '.')
import utils.binvox_rw as binvox_rw
from utils.joint_util import maketree, bfs
from utils.voxel_utils import Cartesian2Voxcoord, bin2sdf, draw_jointmap, center_vox

try:
    import open3d as o3d
    use_o3d=True
except:
    print("Unable to load open3d")
    use_o3d=False

from utils.obj_utils import ObjLoader

class MixamoVoxDataset(data.Dataset):
    def __init__(self, split, configs=None, reduce_motion=True, use_front_faced=False):
        self.data_dir = data_dir = configs['data_dir']
        self.configs = configs
        self.split = split  # [train_models.txt/valid_models.txt/test_models.txt]
        if 'train' in split:
            self.mode = 'train'
        elif 'valid' in split:
            self.mode = 'eval'
        elif 'test' in split:
            self.mode = 'vis'
        self.use_front_faced = use_front_faced

        self.characters_list = open(os.path.join(data_dir, split)).read().splitlines()
        self.motions_list = sorted([motion.split('.binvox')[0] for motion in
                                    os.listdir(os.path.join(data_dir, 'objs/{}'.format(self.characters_list[0])))
                                    if motion.endswith('.binvox') and motion != 'bindpose.binvox'])

        self.sigma = 2.4
        if 'sigma' in configs.keys():
            self.sigma = configs['sigma']
        if configs['overfit']:
            self.characters_list = [self.characters_list[0]]
            self.motions_list = [self.motions_list[0]]

        # reduce number of motions
        # Samba Dancing: ceil(274/15), Warming Up: ceil(95/10)*2, Shoved Reaction With Spin: ceil(45/5)*2,
        def keep_motion(motion):
            if not reduce_motion:
                return True
            motion_name = motion.split('_')[0]
            frame_idx = int(motion.split('_')[-1])
            if self.mode == 'vis':
                return frame_idx == 0
            if motion_name == 'Samba Dancing':
                return frame_idx % 15 == 0
            elif motion_name == 'Warming Up':
                return frame_idx % 10 == 0
            elif motion_name == 'Shoved Reaction With Spin':
                return frame_idx % 5 == 0
            else:  # Back Squat: 18, Drunk Walk Backwards: 13
                return True
        self.motions_list = [motion for motion in self.motions_list
                             if keep_motion(motion)]
        self.num_characters = self.characters_list.__len__()
        self.motions_per_character = self.motions_list.__len__()

        self.n_joint = 22
        self.r = configs['padding']  # 3
        self.dim_ori = configs['dim_ori']  # 82
        self.dim_pad = configs['dim_pad']  # 88

    def __len__(self):
        return self.num_characters * self.motions_per_character

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        character_idx = idx // self.motions_per_character
        motion_idx = idx % self.motions_per_character

        character_name = self.characters_list[character_idx]
        motion_name = self.motions_list[motion_idx]

        data_dir = self.data_dir
        if self.use_front_faced:
            vox_file = os.path.join(data_dir, 'objs_fixed/{}/{}.binvox'.format(character_name, motion_name))
            joint_matrix_file = os.path.join(data_dir,
                                            'transforms_fixed/{}/{}.csv'.format(character_name, motion_name))
        else:
            vox_file = os.path.join(data_dir, 'objs/{}/{}.binvox'.format(character_name, motion_name))
            joint_matrix_file = os.path.join(data_dir,
                                            'transforms/{}/{}.csv'.format(character_name, motion_name))
        # Read binvox
        r, dim_ori, dim_pad = self.r, self.dim_ori, self.dim_pad
        with open(vox_file, 'rb') as f:
            bin_vox = binvox_rw.read_as_3d_array(f)
            meta = {'translate': bin_vox.translate, 'scale': bin_vox.scale, 'dims': bin_vox.dims[0],
                    'character_name': character_name, 'motion_name': motion_name,
                    'mode': self.mode}
        bin_vox_padded = np.zeros((bin_vox.dims[0] + 2 * r, bin_vox.dims[1] + 2 * r, bin_vox.dims[2] + 2 * r), dtype= np.float16)
        bin_vox_padded[r:bin_vox.dims[0] + r, r:bin_vox.dims[1] + r, r:bin_vox.dims[2] + r] = bin_vox.data
        # put the occupied voxels at the center instead of left-top corner
        bin_vox_padded, center_trans = center_vox(bin_vox_padded)
        meta['center_trans'] = np.array(center_trans)
        # convert binary voxels to SDF representation
        sdf_vox_padded = bin2sdf(bin_vox_padded)

        # Relative Coordinate
        JM4x4 = np.concatenate([np.genfromtxt(joint_matrix_file, delimiter=',', dtype='float'), np.array([[0, 0, 0, 1]]*22)],
                               axis=1).reshape(22, 4, 4)
        # IBM4x4 = np.concatenate([np.genfromtxt(inverse_bind_matrix_file, delimiter=',', dtype='float'), np.array([[0, 0, 0, 1]]*22)],
        #                        axis=1).reshape(22, 4, 4)
        # skinweights = np.genfromtxt(skin_file, delimiter=',', dtype='float')
        tree = maketree(22)
        JM4x4_glob = [None] * 22
        bfs(tree, JM4x4, JM4x4_glob)
        JM4x4_glob = np.array(JM4x4_glob)
        JM4x4_glob_p = JM4x4_glob[:, :3, 3]

        target_heatmap = np.zeros((self.n_joint, int(bin_vox.dims[0] + 2 * r), int(bin_vox.dims[1] + 2 * r),
                                  int(bin_vox.dims[2] + 2 * r)), dtype=np.float16)
        for joint_idx in range(self.n_joint):
            heatmap, pos = target_heatmap[joint_idx], JM4x4_glob_p[joint_idx]
            pos = Cartesian2Voxcoord(pos, bin_vox.translate, bin_vox.scale, bin_vox.dims[0])
            pos = (pos[0] - center_trans[0] + r, pos[1] - center_trans[1] + r, pos[2] - center_trans[2] + r)
            pos = np.clip(pos, a_min=0, a_max=dim_pad - 1)
            draw_jointmap(heatmap, pos, sigma=self.sigma)

        return bin_vox_padded, sdf_vox_padded, target_heatmap, JM4x4, meta