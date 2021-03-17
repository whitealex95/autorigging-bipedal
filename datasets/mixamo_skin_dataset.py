import os
from os import listdir
from os.path import join

import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from tqdm import tqdm
try:
    import open3d as o3d
    use_o3d=True
except:
    print("Unable to load open3d")
    use_o3d=False


class MixamoSkinDataset(Dataset):
    def __init__(self, root="data/mixamo", transform=None, pre_transform=None, data_dir="data/mixamo", split='train', vol_geo_dir=None,
                 num_joints=22, center_yaxis=True, preprocess=False, datatype=None, configs=None, test_all=False):
        self.split = split
        self.overfit = split.endswith('overfit')
        self.center_yaxis=center_yaxis
        self.preprocess = preprocess
        self.configs=configs
        self.test_all=test_all

        self.datatype = datatype  # "point"

        self.data_dir = data_dir
        self.obj_dir = join(data_dir, 'objs')  # inference based on objs inside obj_dir
        # self.obj_dir = join(data_dir, 'test_objs')
        self.skin_dir = join(data_dir, 'weights')
        if vol_geo_dir is not None:
            self.vol_geo_dir = join(data_dir, 'volumetric_geodesic')
        else:
            self.vol_geo_dir = vol_geo_dir
        self.num_joints = num_joints

        self.characters_skinweights = {}
        super(MixamoSkinDataset, self).__init__(root, transform, pre_transform)

    @property
    def processed_dir(self):
        if self.datatype is not None:
            return os.path.join(self.data_dir, 'processed') + '_' + self.datatype
        return os.path.join(self.data_dir, 'processed')

    @property
    def raw_dir(self):
        return self.data_dir

    @property
    def raw_file_names(self):
        if self.overfit:
            character, motion = 'aj', 'Samba Dancing_000000'
            raw_objs_list = [join(character, motion+'.obj')]
        else:
            self.characters_list = characters_list = open(os.path.join(self.data_dir, self.split+'.txt')).read().splitlines()
            if self.split == 'test_models' and not self.test_all:
                objs_per_character = [motion for motion
                                      in os.listdir(os.path.join(self.obj_dir, characters_list[0]))
                                      if motion.endswith('000000.obj')]   # use only the first sequence
            else:
                objs_per_character = [motion for motion
                                           in os.listdir(os.path.join(self.obj_dir, characters_list[0]))
                                           if motion.endswith('.obj') and motion != 'bindpose.obj']
            raw_objs_list = []
            for character in characters_list:
                for obj in objs_per_character:
                    raw_objs_list.append(join(character, obj))
        return raw_objs_list
    @property
    def processed_file_names(self):
        return [join(self.split, 'data_{}.pt'.format(i)) for i in range(self.__len__())]

    def __len__(self):
        return len(self.raw_paths)

    def get(self, idx):
        if self.preprocess:
            try:
                data = torch.load(join(self.processed_dir, self.split, 'data_{}.pt'.format(idx)))
                return data
            except:
                print("\n\nFailed to load preprocessed_data\n")
                self.preprocess = False
        data = self.process_single(idx)
        return data

    def process(self):
        if not self.preprocess:
            return
        if not os.path.exists(join(self.processed_dir, self.split)):
            os.makedirs(join(self.processed_dir, self.split))

        for i, motion_obj in tqdm(enumerate(self.raw_file_names)):
            data = self.process_single(i)
            torch.save(data, join(self.processed_dir, self.split, 'data_{}.pt'.format(i)))


    def process_single(self, idx):
        i, motion_obj = idx, self.raw_file_names[idx]
        character_name = motion_obj.split('/')[0]
        motion_name = motion_obj.split('/')[1].split('.obj')[0]
        vol_geo_npy = character_name + '_' + motion_name + '_volumetric_geo.npy'

        if use_o3d:
            mesh = o3d.io.read_triangle_mesh(join(self.obj_dir, motion_obj))
            bp_mesh = o3d.io.read_triangle_mesh(join(self.obj_dir, character_name, 'bindpose.obj'))
            triangles = np.asarray(mesh.triangles)

            vertices = torch.Tensor(np.asarray(mesh.vertices))
            vertex_normals = torch.Tensor(np.asarray(mesh.vertex_normals))
            bp_vertices = torch.FloatTensor(np.asarray(bp_mesh.vertices))

        else:
            raise NotImplementedError

        
        if len(self.characters_skinweights) == 0:
            for cn in tqdm(self.characters_list):
                skinweights = torch.FloatTensor(
                    np.genfromtxt(join(self.skin_dir, cn + '.csv'), delimiter=',', dtype='float'))
                self.characters_skinweights[cn] = skinweights
                
        skinweights=self.characters_skinweights[character_name]
        # y-axis center
        if self.center_yaxis:
            x_max, x_min = vertices[:,0].max().item(), vertices[:,0].min().item()
            z_max, z_min = vertices[:,2].max().item(), vertices[:,2].min().item()
            x_mid, z_mid = (x_max+x_min)/2, (z_max+z_min)/2
            vertices[:, 0] -= x_mid
            vertices[:, 2] -= z_mid

        e=[]
        for f in triangles:
            e += [[f[0], f[1]], [f[1], f[2]], [f[2], f[0]], [f[1], f[0]], [f[0], f[2]], [f[2], f[1]]]
        E = torch.unique(torch.LongTensor(e).T, dim=0)

        vol_geo = torch.Tensor(np.load(join(self.vol_geo_dir, vol_geo_npy)))

        mesh = Data(pos=vertices, normal=vertex_normals, volumetric_geodesic=vol_geo, edge_index=E, skin=skinweights, bindpose=bp_vertices,
                    character_name=character_name, motion_name=motion_name)
        return (mesh, 0, 0, character_name, motion_name)