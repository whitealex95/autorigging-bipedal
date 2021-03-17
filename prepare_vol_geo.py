## for calculating volumetric geodesic distance with rignet format.
## Slowest Part of the pipeline

import os
import numpy as np
import open3d as o3d
import argparse

from utils.tree_utils import TreeNode
from utils.rig_parser import Skel, Info, get_joint_names, maketree
from utils.common_ops import get_bones, calc_surface_geodesic
from utils.compute_volumetric_geodesic import pts2line, calc_pts2bone_visible_mat, calc_geodesic_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--character_idx', type=int, default=-1)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--last', type=int, default=-1)
parser.add_argument('--data_dir', type=str, default='data/mixamo')
parser.add_argument('--joint_log', type=str, default='logs/vox/vox_ori_all_s4.5_HG_mean_stack2_down2_lr3e-4_b4_ce')
args = parser.parse_args()
# pre-calculate volumetric geodesic distance with input mesh & skeleton
print("Calculating volumetric-geodesic distance...")
downsample_skinning = True

joint_name = get_joint_names() # joint name
tree = maketree(22) # hard-coded tree

data_dir = args.data_dir
with open(os.path.join(data_dir, 'test_models.txt')) as test_file:
    characters_list = test_file.read().splitlines()
joint_log = args.joint_log
info_list = os.listdir(os.path.join(joint_log, 'test', characters_list[0]))
motions_list = sorted([info.split('_info')[0] for info in info_list if info.endswith('_info.npy')])
prediction_method = 'pred_heatmap_joint_pos_mask'

for i, character in enumerate(characters_list):
    # if i != args.character_idx:
    #     continue

    for motion in motions_list[args.start:args.last]:
        joint_pos_file = os.path.join(joint_log, 'test', character, '%s_joint.npy' % (motion))
        joint_pos = np.load(joint_pos_file, allow_pickle=True).item()
        joint_result = joint_pos[prediction_method]

        # save skeleton
        pred_skel = Info()
        nodes = []
        for joint_index, joint_pos in enumerate(joint_result):
            nodes.append(TreeNode(name=joint_name[joint_index], pos=joint_pos))

        pred_skel.root = nodes[0]
        for parent, children in enumerate(tree):
            for child in children:
                nodes[parent].children.append(nodes[child])
                nodes[child].parent = nodes[parent]

        # calculate volumetric geodesic distance
        bones, _, _ = get_bones(pred_skel)
        mesh_filename = os.path.join(data_dir, 'objs', character + '/' + motion + '.obj')
        # mesh_filename = os.path.join(data_dir, 'test_objs', character + '_' + motion + '.obj') 
        mesh = o3d.io.read_triangle_mesh(mesh_filename)
        mesh.compute_vertex_normals()
        mesh_v = np.asarray(mesh.vertices)
        surface_geodesic = calc_surface_geodesic(mesh)
        volumetric_geodesic = calc_geodesic_matrix(bones, mesh_v, surface_geodesic, mesh_filename, subsampling=downsample_skinning)
        # save the volumetric_geodesic distance of the prediction inside the joint log folder
        os.makedirs(os.path.join(joint_log, 'volumetric_geodesic_ours_final'), exist_ok=True)

        # save volumetric geodesic distance
        np.save(
                os.path.join(joint_log, 'volumetric_geodesic_ours_final', character + "_" + motion + "_volumetric_geo.npy"),
                volumetric_geodesic
                )
