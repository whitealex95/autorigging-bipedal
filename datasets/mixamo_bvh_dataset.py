import os
import sys

import numpy as np
import torch
import torch.utils.data as data

from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from bvh import Bvh

sys.path.insert(0, '..')
sys.path.insert(0, '.')
from utils.bvh_utils import get_bvh_offsets_and_animation_load_all, get_global_bvh_offsets, get_global_bvh_rotations, get_animated_bvh_joint_positions_single

class MixamoBVHDataset(data.Dataset):
    def __init__(self, split, configs=None, use_front_faced=False, joint_path=None):
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
        self.joint_path = joint_path

        self.characters_list = open(os.path.join(data_dir, split)).read().splitlines()
        # count # of objs
        self.frames_list = sorted([motion.split('.binvox')[0] for motion in
                                    os.listdir(os.path.join(data_dir, 'objs/{}'.format(self.characters_list[0])))
                                    if motion.endswith('.binvox') and motion != 'bindpose.binvox'])
        self.motions_list = []
        for frame_name in self.frames_list:
            motion_name =  '_'.join(frame_name.split('_')[:-1])
            if motion_name not in self.motions_list:
                self.motions_list.append(motion_name)
        if configs['overfit']:
            self.characters_list = [self.characters_list[0]]
            self.frames_list = [self.frames_list[0]]

        # reduce number of motions
        # Samba Dancing: ceil(274/15), Warming Up: ceil(95/10)*2, Shoved Reaction With Spin: ceil(45/5)*2,
        def keep_motion(motion):
            if not configs['reduce_motion']:
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
        self.frames_list = [motion for motion in self.frames_list
                             if keep_motion(motion)]
        self.num_characters = len(self.characters_list)
        self.frames_per_character = len(self.frames_list)

        self.n_joint = 22

        self.mocap_files = [[0 for _ in range(len(self.motions_list))] for _ in range(self.num_characters)]
        self.mocap_info = [[0 for _ in range(len(self.motions_list))] for _ in range(self.num_characters)]

        try:
            self.mocap_files = np.load(os.path.join(data_dir, 'animated/mocap_files_%s.npy'%(self.mode)), allow_pickle=True)
            self.mocap_info = np.load(os.path.join(data_dir, 'animated/mocap_info_%s.npy'%(self.mode)), allow_pickle=True)
        except:
            if self.joint_path is None:
                print("Loading %s mocap files"%(self.mode))
                for character_idx, character_name in tqdm(enumerate(self.characters_list)):
                    for motion_idx, motion_name in enumerate(self.motions_list):
                        bvh_file_name = os.path.join(data_dir, 'animated/{}/{}.bvh'.format(character_name, motion_name))
                        with open(bvh_file_name) as f:
                            mocap = Bvh(f.read())
                            self.mocap_files[character_idx][motion_idx] = mocap
                            offsets_list, rotations_list, root_position_list = get_bvh_offsets_and_animation_load_all(mocap)
                            self.mocap_info[character_idx][motion_idx] = {'offsets': offsets_list, 'rotations': rotations_list,
                                                                            'root_position': root_position_list}
                np.save(os.path.join(data_dir, 'animated/mocap_files_%s'%(self.mode)), self.mocap_files)
                np.save(os.path.join(data_dir, 'animated/mocap_info_%s'%(self.mode)), self.mocap_info)
                print("save complete")



    def __len__(self):
        return self.num_characters * self.frames_per_character

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.joint_path is not None:
            character_idx = idx // self.frames_per_character
            character_name = self.characters_list[character_idx]

            frame_idx = idx % self.frames_per_character
            frame_name = self.frames_list[frame_idx]
            joint_pos_file = os.path.join(self.joint_path, character_name, '%s_joint.npy'%(frame_name))
            joint_pos = np.load(joint_pos_file, allow_pickle=True).item()
            pred_heatmap_joint_pos_mask = joint_pos['pred_heatmap_joint_pos_mask']
            joint_positions = pred_heatmap_joint_pos_mask
            return joint_positions, 0, {'character_name': character_name, 'motion_name': frame_name}


        return self.get_item(self.data_dir, idx, mode=self.mode)

    # @staticmethod
    def get_item(self, data_dir, idx, mode='vis'):
        character_idx = idx // self.frames_per_character
        character_name = self.characters_list[character_idx]

        frame_idx = idx % self.frames_per_character
        frame_name = self.frames_list[frame_idx]
        
        # Samba Dancing_000250 -> Samba Dancing, 250
        motion_name, frame_number = '_'.join(frame_name.split('_')[:-1]), int(frame_name.split('_')[-1])
        motion_idx = self.motions_list.index(motion_name)
        mocap_data = self.mocap_info[character_idx][motion_idx] 
        offsets, rotations, root_position = mocap_data['offsets'][frame_number], mocap_data['rotations'][frame_number], mocap_data['root_position'][frame_number]

        if self.configs['augment_rot_std'] > 0:
            rotations = self.augment_rotation(rotations, self.configs['augment_rot_std'])
        
        glob_rotations = get_global_bvh_rotations(rotations)
        glob_offsets = get_global_bvh_offsets(offsets)
        joint_positions = get_animated_bvh_joint_positions_single(offsets, glob_rotations, root_position)

        if self.configs['jitter_pos_std'] > 0:
            joint_positions = self.jitter_position(joint_positions, self.configs['jitter_pos_std'])

        meta = {
                'character_name': character_name, 'motion_name': frame_name, 'mode': mode,
                'offsets': offsets,
                'rotations': rotations,
                'root_position': root_position
                }
        # ToDo: do some jittering for robust bvh prediction
        return joint_positions, glob_rotations, meta

    def augment_rotation(self, rotations, angle_std):
        rotations_list = []
        for rotation in rotations:  # predefined joint_names
            rot_mat = R.from_matrix(rotation)
            rot_q = rot_mat.as_euler('XYZ', degrees=True)
            
            rot_noise = np.random.normal(0, angle_std, size=(3))  # degrees
            rot_q_aug = rot_q + rot_noise

            rot_q_aug = R.from_euler('XYZ', rot_q_aug, degrees=True)  # should we use 'XYZ', degrees=True?
            rot_aug = rot_q_aug.as_matrix()
            # rot_aug = rot_aug.reshape(1, 3, 3)
            rotations_list.append(rot_aug)
        rotation_matrices = np.stack(rotations_list)
        return rotation_matrices

    def jitter_position(self, positions, pos_std):
        jitter_p = np.random.normal(0, pos_std, size=positions.shape)
        return positions+jitter_p