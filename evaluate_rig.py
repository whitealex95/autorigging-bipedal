# Code for evaluating MPJPE, S-EMD, IOU, etc.
# Some functions are from https://github.com/zhan-xu/RigNet
# Metrics added/modified to match our paper
# In this code, B2B_EMD or b2b_emd represents S-EMD(Skeletal EM Distance) in the paper
# we use --eval_same_skel for our model method while --eval_different_skel is used for rignet 
import torch
from torch_scatter import scatter_mean
import os
import numpy as np
import scipy
import trimesh
from scipy.optimize import linear_sum_assignment
import open3d as o3d
from utils.vis_utils import drawSphere, drawCone
from tqdm import tqdm
import csv
from termcolor import colored
import time
import datetime
import argparse
import pdb
from torch_geometric.nn import fps

def pts2line(pts, lines):
    '''
    Calculate points-to-bone distance. Point to line segment distance refer to
    https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    :param pts: N*3
    :param lines: N*6, where [N,0:3] is the starting position and [N, 3:6] is the ending position
    :return: origins are the neatest projected position of the point on the line.
             ends are the points themselves.
             dist is the distance in between, which is the distance from points to lines.
             Origins and ends will be used for generate rays.
    '''
    l2 = np.sum((lines[:, 3:6] - lines[:, 0:3]) ** 2, axis=1)
    origins = np.zeros((len(pts) * len(lines), 3))
    ends = np.zeros((len(pts) * len(lines), 3))
    dist = np.zeros((len(pts) * len(lines)))
    for l in range(len(lines)):
        if np.abs(l2[l]) < 1e-8:  # for zero-length edges
            origins[l * len(pts):(l + 1) * len(pts)] = lines[l][0:3]
        else:  # for other edges
            t = np.sum((pts - lines[l][0:3][np.newaxis, :]) * (lines[l][3:6] - lines[l][0:3])[np.newaxis, :], axis=1) / \
                l2[l]
            t = np.clip(t, 0, 1)
            t_pos = lines[l][0:3][np.newaxis, :] + t[:, np.newaxis] * (lines[l][3:6] - lines[l][0:3])[np.newaxis, :]
            origins[l * len(pts):(l + 1) * len(pts)] = t_pos
        ends[l * len(pts):(l + 1) * len(pts)] = pts
        dist[l * len(pts):(l + 1) * len(pts)] = np.linalg.norm(
            origins[l * len(pts):(l + 1) * len(pts)] - ends[l * len(pts):(l + 1) * len(pts)], axis=1)
    return origins, ends, dist

def chamfer_distance_with_average(p1, p2):

    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)
    p1 = p1.repeat(p2.size(1), 1, 1)
    p1 = p1.transpose(0, 1)
    p2 = p2.repeat(p1.size(0), 1, 1)
    dist = torch.add(p1, torch.neg(p2))
    dist_norm = torch.norm(dist, 2, dim=2)
    dist1 = torch.min(dist_norm, dim=1)[0]
    dist2 = torch.min(dist_norm, dim=0)[0]
    loss = 0.5 * ((torch.mean(dist1)) + (torch.mean(dist2)))
    return loss


def get_perpend_vec(v):
    max_dim = np.argmax(np.abs(v))
    if max_dim == 0:
        u_0 = np.array([(-2.0 * v[1] - 1.0 * v[2]) / (v[0]+1e-10), 2.0, 1.0])
    elif max_dim == 1:
        u_0 = np.array([1.0, (-1.0 * v[0] - 2.0 * v[2]) / (v[1]+1e-10), 2.0])
    elif max_dim == 2:
        u_0 = np.array([1.0, 2.0, (-1.0 * v[0] - 2.0 * v[1]) / (v[2]+1e-10)])
    u_0 /= np.linalg.norm(u_0)
    return u_0


def cal_perpendicular_dir(p_pos, ch_pos, ray_per_sample):
    dirs = []
    v = (ch_pos - p_pos).squeeze()
    v = v / (np.linalg.norm(v)+1e-10)
    u_0 = get_perpend_vec(v)
    w = np.cross(v, u_0)
    w = w / (np.linalg.norm(w)+1e-10)
    for angle in np.arange(0, 2*np.pi, 2*np.pi/ray_per_sample):
        u = np.cos(angle) * u_0 + np.sin(angle) * w
        u = u / (np.linalg.norm(u)+1e-10)
        dirs.append(u[np.newaxis, :])
    dirs = np.concatenate(dirs, axis=0)
    return dirs


def form_rays(joint_pos_gt, children, ray_per_sample):
    origins_dict = {}
    dirs_dict = {}
    for parent in joint_pos_gt.keys():
        p_pos = np.array(joint_pos_gt[parent])[np.newaxis, :]
        if not parent in origins_dict.keys():
            origins_dict[parent] = p_pos
            dirs_dict[parent] = []

        if parent in children.keys():
            for child in children[parent]:
                c_pos = np.array(joint_pos_gt[child])[np.newaxis, :]
                dir_bone = cal_perpendicular_dir(p_pos, c_pos, ray_per_sample)
                if not child in origins_dict.keys():
                    origins_dict[child] = c_pos
                    dirs_dict[child] = []

                dirs_dict[parent].append(dir_bone)
                dirs_dict[child].append(dir_bone)

    origins = []
    for joint in joint_pos_gt.keys():
        origins.append(origins_dict[joint])
    origins = np.concatenate(origins, axis=0)

    for joint in dirs_dict.keys():
        dirs = dirs_dict[joint]
        dirs_dict[joint] = np.concatenate(dirs, axis=0)

    # tiling
    origins_tile = []
    for i, joint in enumerate(joint_pos_gt.keys()):
        origins_tile.append(np.repeat(origins[i][np.newaxis, :], dirs_dict[joint].shape[0], axis=0))
    origins_tile = np.concatenate(origins_tile, axis=0)
    dirs_tile = []
    for joint in joint_pos_gt.keys():
        dirs_tile.append(dirs_dict[joint])
    dirs_tile = np.concatenate(dirs_tile, axis=0)

    return origins_tile, dirs_tile, dirs_dict, joint_pos_gt.keys(),


def shoot_rays(mesh, origins_tile, ray_dir_tile, dirs_dict, joint_list):
    '''
    shoot rays and record the first hit distance, as well as all vertices on the hit faces.
    :param mesh: input mesh (trimesh)
    :param origins: origin of rays
    :param ray_dir: direction of rays
    :return: all vertices indices on the hit face, the distance of first hit for each ray.
    '''
    RayMeshIntersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, index_tri = RayMeshIntersector.intersects_location(origins_tile, ray_dir_tile + 1e-15)
    locations_per_ray = []
    for i in range(len(ray_dir_tile)):
        locations_per_ray.append(locations[index_ray == i])

    local_shape_diameter = []
    ori_id = 0
    for joint in joint_list:
        dirs = dirs_dict[joint]
        hit_dist = []

        for i in range(dirs.shape[0]):
            ray_id = ori_id + i
            if len(locations_per_ray[ray_id]) > 1:
                closest_hit_id = np.argmin(np.linalg.norm(locations_per_ray[ray_id] - origins_tile[ray_id], axis=1))
                hit_dist.append(np.linalg.norm(locations_per_ray[ray_id][closest_hit_id] - origins_tile[ray_id]))
            elif len(locations_per_ray[ray_id]) == 1:
                hit_dist.append(np.linalg.norm(locations_per_ray[ray_id][0] - origins_tile[ray_id]))

        if len(hit_dist) == 0: # no hit, pick nearby faces
            hit_tri = trimesh.proximity.nearby_faces(mesh, origins_tile[int(ori_id + 0)][np.newaxis, :])[0]
            hit_vertices = mesh.faces[hit_tri].flatten()
            hit_pos = [np.array(mesh.vertices[i])[np.newaxis, :] for i in hit_vertices]
            hit_dist = [np.linalg.norm(hit_pos[i].squeeze() - origins_tile[int(ori_id + 0)]) for i in range(len(hit_pos))]

        ori_id += dirs.shape[0]
        hit_dist = np.array(hit_dist)
        local_shape_diameter.append(np.mean(hit_dist))

    return local_shape_diameter

def read_rig_info(rig_file_path):
    joint_pos = {}
    root = ''
    children = {}
    with open(rig_file_path, 'r') as rig:
        info = rig.read().splitlines()
        for line in info:
            li = line.split(" ")
            if li[-1] == '':
                li = li[:-1]

            if li[0] == 'joints':
                joint_pos[li[1]] = [float(li[2]), float(li[3]), float(li[4])]
            elif li[0] == 'root':
                root = li[1]
            elif li[0] == 'skin':
                continue
            elif li[0] == 'hier':
                if li[1] in children.keys():
                    children[li[1]].append(li[2])
                else:
                    children[li[1]] = [li[2]]

    return joint_pos, root, children

def make_joint_set(joint_pos):
    joint_sets = []
    for joint in joint_pos.keys():
        joint_sets.append(joint_pos[joint])
    joint_sets = np.array(joint_sets)
    return joint_sets

def make_bone_set(joint_pos, children):
    bone_set = []
    for parent in children.keys():
        for child in children[parent]:
            bone = np.concatenate((np.array(joint_pos[parent]), np.array(joint_pos[child])), axis=0)
            bone_set.append(bone)
    bone_set = np.array(bone_set)  # (N-1, 3)
    return bone_set

def chamfer_distance_joint2bone(joint_set_infer, joint_set_gt, bone_set_infer, bone_set_gt):
    # output joint to reference bone
    _, _, dist_o2r = pts2line(joint_set_infer, bone_set_gt)
    joint_num_infer = joint_set_infer.shape[0]
    bone_num_gt = bone_set_gt.shape[0]
    dist_o2r = np.reshape(dist_o2r, (bone_num_gt, joint_num_infer))  # (M-1, N)
    dist_o2r = np.min(dist_o2r, axis=0)

    # reference joint to output bone
    _, _, dist_r2o = pts2line(joint_set_gt, bone_set_infer)
    joint_num_gt = joint_set_gt.shape[0]
    bone_num_infer = bone_set_infer.shape[0]
    dist_r2o = np.reshape(dist_r2o, (bone_num_infer, joint_num_gt))  # (N-1, M)
    dist_r2o = np.min(dist_r2o, axis=0)

    loss = 0.5 * ((np.mean(dist_o2r)) + (np.mean(dist_r2o)))
    return loss

def null(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def b2b(P0, P1, Q0, Q1, eps=1e-14):
    '''
    geomalgorithms.com/a07-_distance.html#dist3D_Segment_to_Segment()
    '''

    u = P1 - P0
    v = Q1 - Q0
    w = P0 - Q0
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    D = a*c - b*b
    sD = D
    tD = D

    if D < eps:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b*e - c*d)
        tN = (a*e - b*d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d+b) < 0.0:
            sN = 0
        elif (-d+b) > a:
            sN = sD
        else:
            sN = -d + b
            sD = a
    sc = 0.0 if (np.abs(sN) < eps) else sN / sD
    tc = 0.0 if (np.abs(tN) < eps) else tN / tD
    dP = w + (sc * u) - (tc * v)
    return np.linalg.norm(dP)

def chamfer_distance_bone2bone(bone_set_infer, bone_set_gt):
    '''
    https://stackoverflow.com/questions/627563/calculating-the-shortest-distance-between-two-lines-line-segments-in-3d
    (second answer)
    '''

    # output bone to reference bone
    dist_o2r = []
    for bone_idx_infer in range(bone_set_infer.shape[0]):
        P = bone_set_infer[bone_idx_infer][:3]
        Q = bone_set_infer[bone_idx_infer][3:]
        dist_o2r_temp = 100000

        for bone_idx_gt in range(bone_set_gt.shape[0]):
            R = bone_set_gt[bone_idx_gt][:3]
            S = bone_set_gt[bone_idx_gt][3:]
            dist_o2r_temp = min(dist_o2r_temp, b2b(P,Q,R,S))
        dist_o2r.append(dist_o2r_temp)
    dist_o2r = np.array(dist_o2r)

    # reference bone to output bone
    dist_r2o = []

    for bone_idx_gt in range(bone_set_gt.shape[0]):
        P = bone_set_gt[bone_idx_gt][:3]
        Q = bone_set_gt[bone_idx_gt][3:]
        dist_r2o_temp = 100000

        for bone_idx_infer in range(bone_set_infer.shape[0]):
            R = bone_set_infer[bone_idx_infer][:3]
            S = bone_set_infer[bone_idx_infer][3:]
            dist_r2o_temp = min(dist_r2o_temp, b2b(P,Q,R,S))
        dist_r2o.append(dist_r2o_temp)

    dist_r2o = np.array(dist_r2o)

    loss = 0.5 * (np.mean(dist_r2o) + np.mean(dist_o2r))
    return loss

def find_maximal_matching(joint_pos_infer, joint_pos_gt):
    infer_pos = []
    gt_pos = []
    for key_infer in joint_pos_infer.keys():
        infer_pos.append(joint_pos_infer[key_infer])
    for key_gt in joint_pos_gt.keys():
        gt_pos.append(joint_pos_gt[key_gt])

    infer_pos = np.array(infer_pos)
    gt_pos = np.array(gt_pos)
    cost = np.sqrt(np.sum((infer_pos[:, np.newaxis, :] - gt_pos[np.newaxis, ...])**2, axis=2))

    row_ind, col_ind = linear_sum_assignment(cost)
    return cost, row_ind, col_ind

def get_local_shape_diameter(mesh_file, joint_pos_gt, children_gt, ray_per_sample):
    mesh = trimesh.load(mesh_file)
    trimesh.repair.fix_normals(mesh)

    origins_tile, ray_dir_tile, dirs_dict, joint_list = form_rays(joint_pos_gt, children_gt, ray_per_sample)
    local_shape_diameter = shoot_rays(mesh, origins_tile, ray_dir_tile, dirs_dict, joint_list)
    return np.array(local_shape_diameter)

def sampling_point_cloud_fps(bone, per_bone_sample=50, total_num_point=300):
    point_list = []
    for i in range(len(bone)):
        start_pos = bone[i][:3]
        end_pos = bone[i][3:]
        for idx in range(1, per_bone_sample+1):
            point_list.append(start_pos * idx / (per_bone_sample+2) + end_pos * (per_bone_sample+2-idx) / (per_bone_sample+2))

    for i in range(len(bone)):
        point_list.append(bone[i][:3])
        point_list.append(bone[i][3:])

    point_cloud = np.array(point_list)
    point_cloud = torch.from_numpy(point_cloud).cuda() # (52*B, 3)
    idx = fps(point_cloud, None, ratio=total_num_point / ((per_bone_sample+2)*len(bone)))
    points = point_cloud[idx]
    return points

def sampling_point_cloud(bone, total_num_point=300):
    point_list = []
    num_bone = len(bone)
    per_bone_sample = int(total_num_point / num_bone)

    for i in range(len(bone)):
        start_pos = bone[i][:3]
        end_pos = bone[i][3:]
        for idx in range(1, per_bone_sample + 1):
            point_list.append(start_pos * idx / (per_bone_sample + 2) + end_pos * (per_bone_sample + 2 - idx) / (per_bone_sample + 2))

    res = total_num_point - per_bone_sample * num_bone
    for i in range(res):
        selected_bone = np.random.randint(num_bone)
        start_pos = bone[selected_bone][:3]
        end_pos = bone[selected_bone][3:]
        rand_ratio = np.random.rand()
        point_list.append(start_pos * rand_ratio + end_pos * (1 - rand_ratio))

    point_cloud = np.array(point_list)
    return point_cloud

def eval_different_skel(args, dataset_dir, tolerance_level=1.0, ray_per_sample=14, local_shape_diameter_debug=False):
    test_obj_dir = 'test_objs'
    gt_rig_dir = 'test_rigs'
    # test list
    with open(os.path.join(dataset_dir, 'test.txt')) as test_file:
        test_list = test_file.read().splitlines()

    csv_file = open(os.path.join(dataset_dir, args.target_rig_dir, 'results.csv'), 'w')
    writer = csv.writer(csv_file, delimiter=',', quotechar='"')

    write_list = ['id', 'cd_j2j', 'cd_j2b', 'cd_b2b', 'IoU', 'precision', 'recall', 'j2j', 'b2b_emd']
    writer.writerow(write_list)
    total = {}
    for item in write_list:
        if item == 'id':
            continue
        total[item] = 0.0

    start_time = time.time()

    for obj_name in tqdm(test_list):
        mesh_file = obj_name + '.obj'
        output_rig_file = obj_name + '_rig.txt'
        gt_rig_file = obj_name + '.txt'

        joint_pos_infer, root_infer, children_infer = read_rig_info(os.path.join(dataset_dir, args.target_rig_dir, output_rig_file))
        joint_pos_gt, root_gt, children_gt = read_rig_info(os.path.join(dataset_dir, gt_rig_dir, gt_rig_file))

        joint_set_infer = make_joint_set(joint_pos_infer) # (N, 3)
        joint_set_gt = make_joint_set(joint_pos_gt) # (M, 3)

        bone_set_infer = make_bone_set(joint_pos_infer, children_infer) # (N-1, 3)
        bone_set_gt = make_bone_set(joint_pos_gt, children_gt) # (M-1, 3)

        # CD-J2J
        cd_j2j = chamfer_distance_with_average(torch.from_numpy(joint_set_infer).unsqueeze(0), torch.from_numpy(joint_set_gt).unsqueeze(0))
        cd_j2j = cd_j2j.detach().cpu().numpy()
        #print("CD_J2J: " + str(cd_j2j.cpu().numpy()))

        # CD-J2B
        cd_j2b = chamfer_distance_joint2bone(joint_set_infer, joint_set_gt, bone_set_infer, bone_set_gt)
        #print("CD_J2B: " + str(cd_j2b))

        # CD-B2B
        cd_b2b = chamfer_distance_bone2bone(bone_set_infer, bone_set_gt)
        #print("CD_B2B: " + str(cd_b2b))

        # IoU, Precision, Recall
        cost, infer_ind, gt_ind = find_maximal_matching(joint_pos_infer, joint_pos_gt)
        local_shape_diameter = get_local_shape_diameter(os.path.join(dataset_dir, test_obj_dir, mesh_file),
                                                        joint_pos_gt, children_gt, ray_per_sample)
        local_shape_diameter = tolerance_level * local_shape_diameter

        intersection = 0
        for b in cost[infer_ind, gt_ind] < local_shape_diameter[gt_ind]:
            if b:
                intersection += 2
        union = len(joint_pos_infer.keys()) + len(joint_pos_gt.keys())
        IoU = intersection / union
        #print("IoU: " + str(IoU))

        # visualization of local shape diameter for debugging
        if local_shape_diameter_debug:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            ctr = vis.get_view_control()

            mesh = o3d.io.read_triangle_mesh(os.path.join(dataset_dir, test_obj_dir, mesh_file))
            mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
            vis.add_geometry(mesh_ls)

            for i, joint in enumerate(joint_pos_gt.keys()):
                vis.add_geometry(drawSphere(np.array(joint_pos_gt[joint]), 0.03, color=[7*local_shape_diameter[i], 0, 0]))
            vis.run()
            vis.destroy_window()

        # Precision
        precision = 0
        ref_idx_nearest_to_matched_output = np.argmin(cost[infer_ind, :], axis=1)
        for b in cost[infer_ind, ref_idx_nearest_to_matched_output] < local_shape_diameter[ref_idx_nearest_to_matched_output]:
            if b:
                precision += 1
        precision = precision / len(joint_pos_infer.keys())
        #print("Precision: " + str(precision))

        # Recall
        recall = 0
        output_idx_nearest_to_matched_ref = np.argmin(cost[:, gt_ind], axis=0)
        for b in cost[output_idx_nearest_to_matched_ref, gt_ind] < local_shape_diameter[gt_ind]:
            if b:
                recall += 1
        recall = recall / len(joint_pos_gt.keys())
        #print("Recall: " + str(recall))

        # J2J
        infer_pos = []
        gt_pos = []
        for key_infer in joint_pos_infer.keys():
            infer_pos.append(joint_pos_infer[key_infer])
        for key_gt in joint_pos_gt.keys():
            gt_pos.append(joint_pos_gt[key_gt])

        infer_pos = np.array(infer_pos)
        gt_pos = np.array(gt_pos)

        j2j_diff = infer_pos[infer_ind] - gt_pos[gt_ind]
        j2j = np.mean(np.linalg.norm(j2j_diff, axis=1))

        # B2B_EMD
        b2b_debug = False
        if args.sampling == 'fps':
            infer_skel_points = sampling_point_cloud_fps(bone_set_infer, per_bone_sample=50).cpu().numpy()
            gt_skel_points = sampling_point_cloud_fps(bone_set_gt, per_bone_sample=50).cpu().numpy()
        elif args.sampling == 'density':
            infer_skel_points = sampling_point_cloud(bone_set_infer)
            gt_skel_points = sampling_point_cloud(bone_set_gt)
        else:
            raise NotImplementedError

        assert (infer_skel_points.shape[0] == gt_skel_points.shape[0])

        if b2b_debug:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            ctr = vis.get_view_control()

            mesh = o3d.io.read_triangle_mesh(os.path.join(dataset_dir, test_obj_dir, mesh_file))
            mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
            vis.add_geometry(mesh_ls)

            gt_skel_points_test = gt_skel_points
            for i, joint in enumerate(gt_skel_points_test):
                vis.add_geometry(
                    drawSphere(np.array(gt_skel_points_test[i]), 0.005, color=[1, 0, 0]))
            vis.run()
            vis.destroy_window()

        skel_points_dist = np.sqrt(np.sum((infer_skel_points[:, np.newaxis, :] - gt_skel_points[np.newaxis, ...]) ** 2, axis=2))
        infer_skel_ind, gt_skel_ind = linear_sum_assignment(skel_points_dist)
        b2b_emd = np.mean(skel_points_dist[infer_skel_ind, gt_skel_ind])

        writer.writerow([obj_name, cd_j2j, cd_j2b, cd_b2b, IoU, precision, recall, j2j, b2b_emd])
        csv_file.flush()

        total['cd_j2j'] += cd_j2j
        total['cd_j2b'] += cd_j2b
        total['cd_b2b'] += cd_b2b
        total['IoU'] += IoU
        total['precision'] += precision
        total['recall'] += recall
        total['j2j'] += j2j
        total['b2b_emd'] += b2b_emd

    writer.writerow(['mean', total['cd_j2j']/len(test_list), total['cd_j2b']/len(test_list),
                    total['cd_b2b'] / len(test_list), total['IoU']/len(test_list), total['precision']/len(test_list),
                    total['recall'] / len(test_list), total['j2j'] / len(test_list), total['b2b_emd'] / len(test_list)])
    csv_file.close()

    for item in total.keys():
        print(colored('Average ' + item + ': %f' % (total[item]/len(test_list)), 'grey', 'on_yellow'))

    with open(os.path.join(dataset_dir, args.target_rig_dir, 'results_summary.txt'), 'w') as summary:
        summary.write(args.target_rig_dir + ' results summary\n\n')
        for item in total.keys():
            summary.write('Average ' + item + ': %.8f \n' % (total[item] / len(test_list)))

    print(colored("Testing done", 'white', 'on_magenta'))
    print(colored("Total time: " + str(datetime.timedelta(seconds=time.time() - start_time)), 'white', 'on_magenta'))

def eval_same_skel(args, dataset_dir, tolerance_level=1.0, ray_per_sample=14, local_shape_diameter_debug=False):
    test_obj_dir = 'test_objs'
    gt_rig_dir = 'test_rigs'

    joint_names = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm',
                   'RightShoulder', 'RightArm', 'RightForeArm', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
                   'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
                   'LeftHand', 'RightHand']

    # test list
    with open(os.path.join(dataset_dir, 'test.txt')) as test_file:
        test_list = test_file.read().splitlines()

    csv_file = open(os.path.join(dataset_dir, args.target_rig_dir, 'results.csv'), 'w')
    writer = csv.writer(csv_file, delimiter=',', quotechar='"')

    write_list = ['id', 'j2j', 'j2b', 'b2b', 'IoU', 'precision', 'recall', 'b2b_emd']
    writer.writerow(write_list)
    total = {}
    for item in write_list:
        if item == 'id':
            continue
        total[item] = 0.0

    start_time = time.time()

    for obj_name in tqdm(test_list):
        mesh_file = obj_name + '.obj'
        output_rig_file = obj_name + '_rig.txt'
        gt_rig_file = obj_name + '.txt'

        joint_pos_infer, root_infer, children_infer = read_rig_info(
            os.path.join(dataset_dir, args.target_rig_dir, output_rig_file))
        joint_pos_gt, root_gt, children_gt = read_rig_info(os.path.join(dataset_dir, gt_rig_dir, gt_rig_file))

        joint_set_infer = make_joint_set(joint_pos_infer)  # (J, 3)
        joint_set_gt = make_joint_set(joint_pos_gt)  # (J, 3)

        bone_set_infer = make_bone_set(joint_pos_infer, children_infer)  # (J-1, 3)
        bone_set_gt = make_bone_set(joint_pos_gt, children_gt)  # (J-1, 3)

        # J2J
        diff = joint_set_infer - joint_set_gt
        dist = np.sqrt(np.sum(diff**2, axis=1))
        j2j_dist = np.mean(dist)

        # J2B
        cd_j2b = chamfer_distance_joint2bone(joint_set_infer, joint_set_gt, bone_set_infer, bone_set_gt)

        def bone_idx(joint_pos, children):
            li = list(joint_pos.keys())
            ret = [[] for i in range(22)]
            bone_set = []
            for parent in children.keys():
                for child in children[parent]:
                    bone = np.concatenate((np.array(joint_pos[parent]), np.array(joint_pos[child])), axis=0)
                    ret[li.index(parent)].append(len(bone_set))
                    ret[li.index(child)].append(len(bone_set))
                    bone_set.append(bone)
            return ret

        b_idx = bone_idx(joint_pos_gt, children_gt)

        # output to reference
        dist_o2r = 0.0
        for i in range(joint_set_infer.shape[0]):
            min_j2b_dist_o2r = 100000
            for bone in b_idx[i]:
                _, _, dist_o2r_temp = pts2line(joint_set_infer[i][np.newaxis, ...], bone_set_gt[bone][np.newaxis, ...])
                min_j2b_dist_o2r = min(min_j2b_dist_o2r, dist_o2r_temp[0])
            dist_o2r += min_j2b_dist_o2r
        dist_o2r /= joint_set_infer.shape[0]

        # reference to output
        dist_r2o = 0.0
        for i in range(joint_set_gt.shape[0]):
            min_j2b_dist_r2o = 100000
            for bone in b_idx[i]:
                _, _, dist_r2o_temp = pts2line(joint_set_gt[i][np.newaxis, ...], bone_set_infer[bone][np.newaxis, ...])
                min_j2b_dist_r2o = min(min_j2b_dist_r2o, dist_r2o_temp[0])
            dist_r2o += min_j2b_dist_r2o

        dist_r2o /= joint_set_gt.shape[0]
        j2b_dist = 0.5 * (dist_o2r + dist_r2o)

        # B2B
        b2b_dist = 0.0
        for i in range(bone_set_infer.shape[0]):
            b2b_dist += b2b(
                bone_set_infer[i][:3], bone_set_infer[i][3:],
                bone_set_gt[i][:3], bone_set_gt[i][3:],
                )
        b2b_dist /= bone_set_infer.shape[0]

        # IoU, Precision, Recall
        local_shape_diameter = get_local_shape_diameter(os.path.join(dataset_dir, test_obj_dir, mesh_file),
                                                        joint_pos_gt, children_gt, ray_per_sample)
        local_shape_diameter = tolerance_level * local_shape_diameter

        intersection = 0
        for i in range(joint_set_infer.shape[0]):
            if dist[i] < local_shape_diameter[i]:
                intersection += 2

        union = len(joint_pos_infer.keys()) + len(joint_pos_gt.keys())
        IoU = intersection / union
        # print("IoU: " + str(IoU))

        # visualization of local shape diameter for debugging
        if local_shape_diameter_debug:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            ctr = vis.get_view_control()

            mesh = o3d.io.read_triangle_mesh(os.path.join(dataset_dir, test_obj_dir, mesh_file))
            mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
            vis.add_geometry(mesh_ls)

            for i, joint in enumerate(joint_pos_gt.keys()):
                vis.add_geometry(
                    drawSphere(np.array(joint_pos_gt[joint]), 0.03, color=[7 * local_shape_diameter[i], 0, 0]))
            vis.run()
            vis.destroy_window()


        # Precision (maybe no meaning?)
        all_diff = joint_set_infer[:, np.newaxis, :] - joint_set_gt[np.newaxis, ...]
        all_dist = np.sqrt(np.sum(all_diff**2, axis=2)) # (J, J)
        precision = 0
        ref_idx_nearest_to_output = np.argmin(all_dist, axis=1)
        for b in all_dist[np.arange(len(joint_pos_infer.keys())), ref_idx_nearest_to_output] < local_shape_diameter[ref_idx_nearest_to_output]:
            if b:
                precision += 1

        precision = precision / len(joint_pos_infer.keys())
        # print("Precision: " + str(precision))

        # Recall (maybe no meaning?)
        recall = 0
        output_idx_nearest_to_ref = np.argmin(all_dist, axis=0)
        for b in all_dist[output_idx_nearest_to_ref, np.arange(len(joint_pos_gt.keys()))] < local_shape_diameter[np.arange(len(joint_pos_gt.keys()))]:
            if b:
                recall += 1
        recall = recall / len(joint_pos_gt.keys())
        # print("Recall: " + str(recall))

        # B2B_EMD
        b2b_debug = False
        if args.sampling == 'fps':
            infer_skel_points = sampling_point_cloud_fps(bone_set_infer, per_bone_sample=50).cpu().numpy()
            gt_skel_points = sampling_point_cloud_fps(bone_set_gt, per_bone_sample=50).cpu().numpy()
        elif args.sampling == 'density':
            infer_skel_points = sampling_point_cloud(bone_set_infer)
            gt_skel_points = sampling_point_cloud(bone_set_gt)
        else:
            raise NotImplementedError

        assert(infer_skel_points.shape[0]==gt_skel_points.shape[0])

        if b2b_debug:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            ctr = vis.get_view_control()

            mesh = o3d.io.read_triangle_mesh(os.path.join(dataset_dir, test_obj_dir, mesh_file))
            mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
            vis.add_geometry(mesh_ls)

            gt_skel_points_test = gt_skel_points
            for i, joint in enumerate(gt_skel_points_test):
                vis.add_geometry(
                    drawSphere(np.array(gt_skel_points_test[i]), 0.005, color=[1, 0, 0]))
            vis.run()
            vis.destroy_window()

        skel_points_dist = np.sqrt(
            np.sum((infer_skel_points[:, np.newaxis, :] - gt_skel_points[np.newaxis, ...]) ** 2, axis=2))
        infer_skel_ind, gt_skel_ind = linear_sum_assignment(skel_points_dist)
        b2b_emd = np.mean(skel_points_dist[infer_skel_ind, gt_skel_ind])

        writer.writerow([obj_name, j2j_dist, j2b_dist, b2b_dist, IoU, precision, recall, b2b_emd])
        csv_file.flush()

        total['j2j'] += j2j_dist
        total['j2b'] += j2b_dist
        total['b2b'] += b2b_dist
        total['IoU'] += IoU
        total['precision'] += precision
        total['recall'] += recall
        total['b2b_emd'] += b2b_emd

    writer.writerow(['mean', total['j2j'] / len(test_list), total['j2b'] / len(test_list),
                     total['b2b'] / len(test_list), total['IoU'] / len(test_list),
                     total['precision'] / len(test_list),
                     total['recall'] / len(test_list),
                     total['b2b_emd'] / len(test_list)])
    csv_file.close()

    for item in total.keys():
        print(colored('Average ' + item + ': %f' % (total[item] / len(test_list)), 'grey', 'on_yellow'))

    with open(os.path.join(dataset_dir, args.target_rig_dir, 'results_summary.txt'), 'w') as summary:
        summary.write(args.target_rig_dir + ' results summary\n\n')
        for item in total.keys():
            summary.write('Average ' + item + ': %.8f \n' % (total[item] / len(test_list)))

    print(colored("Testing done", 'white', 'on_magenta'))
    print(colored("Total time: " + str(datetime.timedelta(seconds=time.time() - start_time)), 'white', 'on_magenta'))

if __name__ == '__main__':
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_rig_dir', default='no_finetune_threshold=1.5e-5')
    parser.add_argument('--same_skeleton', action='store_true')
    parser.add_argument('--tolerance', type=float, default=1.0)
    parser.add_argument('--sampling', default='density', help='density, fps, uniform')
    parser.add_argument('--dataset_dir', default='data/mixamo_for_rignet/')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    if args.same_skeleton:
        eval_same_skel(args, dataset_dir, args.tolerance)
    else:
        eval_different_skel(args, dataset_dir, args.tolerance)
