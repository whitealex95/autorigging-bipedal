import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh import Bvh

from utils.joint_util import joint_names, maketree

joint_tree = maketree(22)
parent_tree = [-1] * len(joint_tree)
for parent in range(len(joint_tree)):
    for child in joint_tree[parent]:
        parent_tree[child] = parent

def get_bvh_offsets_and_animation(file_name, frame_number):
    with open(file_name) as f:
        mocap = Bvh(f.read())

    mocap_joint_names = mocap.get_joints_names()
    mocap_joint_prefix = mocap_joint_names[0].split('Hips')[0]

    rotation_matrices = np.empty((1, 3, 3), dtype=float)
    offsets = np.empty((1, 3), dtype=float)
    root_position = mocap.frame_joint_channels(frame_number, mocap_joint_names[0],
                                               ['Xposition', 'Yposition', 'Zposition'])
    for name in joint_names:  # predefined joint_names
        mocap_joint_name = mocap_joint_prefix + name
        euler = mocap.frame_joint_channels(frame_number, mocap_joint_name, ['Xrotation', 'Yrotation', 'Zrotation'])
        #     print(name, euler)
        # rot = R.from_euler('xyz', euler, degrees=False)  # Previous models are trained for wrong rotations
        rot = R.from_euler('XYZ', euler, degrees=True)  # should we use 'XYZ', degrees=True?
        rot = np.array(rot.as_matrix())
        rot = rot.reshape(1, 3, 3)
        rotation_matrices = np.append(rotation_matrices, rot, axis=0)

        offset = np.array(mocap.joint_offset(mocap_joint_name))
        offsets = np.append(offsets, offset.reshape(1, 3), axis=0)

        # print('{}:\nRotationMatrix:\n'.format(name), rot[0], '\nOffset: ', offset.reshape(3, ))

    offsets = np.delete(offsets, [0, 0], axis=0)
    rotation_matrices = np.delete(rotation_matrices, [0, 0, 0], axis=0)
    return offsets, rotation_matrices, root_position  #[3], [J, 3], [J, 3]

def get_bvh_offsets_and_animation_load_all(mocap):
    mocap_joint_names = mocap.get_joints_names()
    mocap_joint_prefix = mocap_joint_names[0].split('Hips')[0]

    offsets_list = []
    rotation_matrices_list = []
    root_position_list = []
    for frame_number in  range(mocap.nframes):
        rotation_matrices = np.empty((1, 3, 3), dtype=float)
        offsets = np.empty((1, 3), dtype=float)
        root_position = mocap.frame_joint_channels(frame_number, mocap_joint_names[0],
                                                ['Xposition', 'Yposition', 'Zposition'])
        for name in joint_names:  # predefined joint_names
            mocap_joint_name = mocap_joint_prefix + name
            euler = mocap.frame_joint_channels(frame_number, mocap_joint_name, ['Xrotation', 'Yrotation', 'Zrotation'])
            #     print(name, euler)
            # rot = R.from_euler('xyz', euler, degrees=False)  # Previous models are trained for wrong rotations
            rot = R.from_euler('XYZ', euler, degrees=True)  # should we use 'XYZ', degrees=True?
            rot = np.array(rot.as_matrix())
            rot = rot.reshape(1, 3, 3)
            rotation_matrices = np.append(rotation_matrices, rot, axis=0)

            offset = np.array(mocap.joint_offset(mocap_joint_name))
            offsets = np.append(offsets, offset.reshape(1, 3), axis=0)

            # print('{}:\nRotationMatrix:\n'.format(name), rot[0], '\nOffset: ', offset.reshape(3, ))

        offsets = np.delete(offsets, [0, 0], axis=0)
        rotation_matrices = np.delete(rotation_matrices, [0, 0, 0], axis=0)

        offsets_list.append(offsets)
        rotation_matrices_list.append(rotation_matrices)
        root_position_list.append(root_position)

    return offsets_list, rotation_matrices_list, root_position_list

def get_bvh_offsets_and_animation_loaded(mocap, frame_number):
    mocap_joint_names = mocap.get_joints_names()
    mocap_joint_prefix = mocap_joint_names[0].split('Hips')[0]

    rotation_matrices = np.empty((1, 3, 3), dtype=float)
    offsets = np.empty((1, 3), dtype=float)
    root_position = mocap.frame_joint_channels(frame_number, mocap_joint_names[0],
                                               ['Xposition', 'Yposition', 'Zposition'])
    for name in joint_names:  # predefined joint_names
        mocap_joint_name = mocap_joint_prefix + name
        euler = mocap.frame_joint_channels(frame_number, mocap_joint_name, ['Xrotation', 'Yrotation', 'Zrotation'])
        #     print(name, euler)
        # rot = R.from_euler('xyz', euler, degrees=False)  # Previous models are trained for wrong rotations
        rot = R.from_euler('XYZ', euler, degrees=True)  # should we use 'XYZ', degrees=True?
        rot = np.array(rot.as_matrix())
        rot = rot.reshape(1, 3, 3)
        rotation_matrices = np.append(rotation_matrices, rot, axis=0)

        offset = np.array(mocap.joint_offset(mocap_joint_name))
        offsets = np.append(offsets, offset.reshape(1, 3), axis=0)

        # print('{}:\nRotationMatrix:\n'.format(name), rot[0], '\nOffset: ', offset.reshape(3, ))

    offsets = np.delete(offsets, [0, 0], axis=0)
    rotation_matrices = np.delete(rotation_matrices, [0, 0, 0], axis=0)
    return offsets, rotation_matrices, root_position


# Single
def get_global_bvh_offsets(offsets):
    global_offsets = np.stack([offsets[0]] + [np.zeros_like(offsets[0])] * (offsets.shape[0]-1))
    for parent_idx, child_list in enumerate(joint_tree):
        for child_idx in child_list:
            global_offsets[child_idx] = global_offsets[parent_idx] + offsets[child_idx]
    return global_offsets

# Single
def get_local_bvh_offsets(global_offsets):
    offsets = np.copy(global_offsets)
    for parent_idx, child_list in enumerate(joint_tree):
        for child_idx in child_list:
            offsets[child_idx] -= global_offsets[parent_idx]
    return offsets

# Single
def get_global_bvh_offsets2(offsets):  # same as the above
    global_offsets = np.copy(offsets)
    for parent_idx, child_list in enumerate(joint_tree):
        for child_idx in child_list:
            global_offsets[child_idx] += global_offsets[parent_idx]
    return global_offsets


# Single, numpy
def get_global_bvh_rotations(rotations):
    global_rotations = np.stack([rotations[0]] + [np.zeros_like(rotations[0])] * (rotations.shape[0]-1))
    for parent_idx, child_list in enumerate(joint_tree):
        for child_idx in child_list:
            global_rotations[child_idx] = np.matmul(global_rotations[parent_idx], rotations[child_idx])
    return global_rotations

# Batch, torch
def get_global_bvh_rotations_torch(rotations:torch.Tensor):
    # [B, J, 3, 3]
    global_rotations = torch.zeros_like(rotations)
    global_rotations[:, 0, :] = rotations[:, 0, :]
    for parent_idx, child_list in enumerate(joint_tree):
        for child_idx in child_list:
            global_rotations[:, child_idx, :] = torch.matmul(global_rotations[:, parent_idx, :], rotations[:, child_idx, :])
    return global_rotations


# Single, numpy
def get_local_bvh_rotations(global_rotations):
    local_rotations = np.stack([global_rotations[0]] + [np.zeros_like(global_rotations[0])] * (global_rotations.shape[0]-1))
    for parent_idx, child_list in enumerate(joint_tree):
        for child_idx in child_list:
            local_rotations[child_idx] = np.matmul(np.linalg.inv(global_rotations[parent_idx]), global_rotations[child_idx])
    return local_rotations


def get_animated_bvh_joint_positions(offsets, global_rotations, root_position=None):
    '''
    offsets: [B, J, 3]
    global_rotations: [B, J, 3, 3]
    root_position: [B, 3]

    return: [B, J, 3]
    '''

    bvh_joint_positions = torch.empty(offsets.shape)
    if root_position is None:
        root_position = offsets[:, 0, :]
    bvh_joint_positions[:, 0, :] = root_position

    offsets = offsets.reshape([-1, offsets.shape[-2], offsets.shape[-1], 1])  # [B, J, 3, 1]
    for i, pi in enumerate(parent_tree):
        if pi == -1:  # parent idx
            assert i == 0
            continue
        bvh_joint_positions[:, i, :] = torch.matmul(global_rotations[:, pi, :, :], offsets[:, i, :, :]).squeeze(-1)
        bvh_joint_positions[:, i, :] += bvh_joint_positions[:, pi, :]
    # print(bvh_joint_positions)
    return bvh_joint_positions





# Single, numpy
def get_animated_bvh_joint_positions_single(offsets, global_rotations, root_position):
    bvh_joint_positions = np.stack([root_position] + [np.zeros_like(offsets[0])] * (offsets.shape[0] - 1))
    for parent_idx, child_list in enumerate(joint_tree):
        for child_idx in child_list:
            bvh_joint_positions[child_idx] = bvh_joint_positions[parent_idx] \
                                             + np.matmul(global_rotations[parent_idx], offsets[child_idx])
    return bvh_joint_positions
#
# # Batch, torch
# def get_animated_bvh_joint_positions_batch(offsets, rotations, root_position, rotation_type='glob'):
#     result = torch.empty(rotations.shape[:-1] + (3,), device=rotations.device)
#     result[:, 0, :] = root_position
#     global_rotations = rotations
#     for parent_idx, child_list in enumerate(joint_tree):
#         for child_idx in child_list:
#             if rotation_type == 'rel':
#
#
#             bvh_joint_positions[child_idx] = bvh_joint_positions[parent_idx] \
#                                              + np.matmul(global_rotations[parent_idx], offsets[child_idx])
#     return bvh_joint_positions
#

# Batch version, torch
def compute_bindpose_from_bvh_animation(global_joint_positions, global_rotations, bindpose_root_offset=None):
    if bindpose_root_offset is not None:
        root_offset = bindpose_root_offset
    else:
        root_offset = global_joint_positions[:, 0, :]
    offsets = torch.stack([root_offset] + [torch.zeros_like(global_joint_positions[:,0,:])] * (
            global_joint_positions.shape[1] - 1), dim=1)
    global_offsets = offsets.clone()
    for parent_idx, child_list in enumerate(joint_tree):
        for child_idx in child_list:
            global_joint_positions_offset = global_joint_positions[:,child_idx,...] - global_joint_positions[:,parent_idx,...]
            offsets[:,child_idx,...] = torch.bmm(torch.inverse(global_rotations[:,parent_idx,...]),
                                              global_joint_positions_offset.unsqueeze(-1)).squeeze(-1)
            global_offsets[:,child_idx,...] = global_offsets[:,parent_idx,...] + offsets[:,child_idx,...]
    return global_offsets


# Single version
def compute_bindpose_from_bvh_animation_single(global_joint_positions, global_rotations, bindpose_root_offset=None):
    if bindpose_root_offset is not None:
        root_offset = bindpose_root_offset
    else:
        root_offset = global_joint_positions[0]
    offsets = np.zeros_like(global_joint_positions)
    global_offsets = np.stack(
        [root_offset] + [np.zeros_like(global_joint_positions[0])] * (global_joint_positions.shape[0] - 1))

    for parent_idx, child_list in enumerate(joint_tree):
        for child_idx in child_list:
            global_joint_positions_offset = global_joint_positions[child_idx] - global_joint_positions[parent_idx]
            offsets[child_idx] = np.matmul(np.linalg.inv(global_rotations[parent_idx]), global_joint_positions_offset)
            global_offsets[child_idx] = global_offsets[parent_idx] + offsets[child_idx]
    return global_offsets


# def compute_bindpose_from_bvh_animation_single(global_joint_positions, global_rotations):
#     if isinstance(global_rotations, torch.Tensor):
#         if len(global_joint_positions.shape)==3:  # batch
#             res = []
#             for idx in range(global_joint_positions.shape[0]):
#                 res.append(compute_bindpose_from_bvh_animation_single(global_joint_positions[idx], global_rotations[idx]))
#             return torch.stack(res)
#
#         offsets = torch.stack([global_joint_positions[0]] + [torch.zeros_like(global_joint_positions[0])] * (
#                 global_joint_positions.shape[0] - 1))
#         global_offsets = offsets.clone()
#         for parent_idx, child_list in enumerate(joint_tree):
#             for child_idx in child_list:
#                 global_joint_positions_offset = global_joint_positions[child_idx] - global_joint_positions[parent_idx]
#                 offsets[child_idx] = torch.matmul(torch.inverse(global_rotations[parent_idx]),
#                                                global_joint_positions_offset)
#                 global_offsets[child_idx] = global_offsets[parent_idx] + offsets[child_idx]
#         return global_offsets

    offsets = np.stack([global_joint_positions[0]] + [np.zeros_like(global_joint_positions[0])] * (global_joint_positions.shape[0]-1))
    global_offsets = np.copy(offsets)
    for parent_idx, child_list in enumerate(joint_tree):
        for child_idx in child_list:
            global_joint_positions_offset = global_joint_positions[child_idx] - global_joint_positions[parent_idx]
            offsets[child_idx] = np.matmul(np.linalg.inv(global_rotations[parent_idx]), global_joint_positions_offset)
            global_offsets[child_idx] = global_offsets[parent_idx] + offsets[child_idx]
    return global_offsets

def bvh_to_JMBM(global_joint_pos, global_rotation_matrices, bindpose_root_offset=None, JM_type='rel'):
    if bindpose_root_offset is None:
        bindpose_root_offset = global_joint_pos[0]
    # Input direct network outputs
    bBM4x4_p = compute_bindpose_from_bvh_animation_single(global_joint_pos, global_rotation_matrices, bindpose_root_offset)
    bJM4x4_glob_p = global_joint_pos
    bJM4x4_glob_rot = global_rotation_matrices
    bBM4x4_rot = np.stack([np.eye(3)]*len(bJM4x4_glob_rot))

    bBM4x4 = np.concatenate([np.concatenate([bBM4x4_rot, bBM4x4_p[:,:,None]], axis=2),
                          np.stack([np.array([[0,0,0,1]])]*22)], axis=1)
    bJM4x4_glob = np.concatenate([np.concatenate([bJM4x4_glob_rot, bJM4x4_glob_p[:,:,None]], axis=2),
                          np.stack([np.array([[0,0,0,1]])]*22)], axis=1)
    if JM_type == 'rel':
        raise NotImplemented
    return bJM4x4_glob, bBM4x4
