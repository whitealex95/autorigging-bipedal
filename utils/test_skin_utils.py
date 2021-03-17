import sys
import os
import random
import datetime
import shutil
import yaml
from configargparse import ArgumentParser, YAMLConfigFileParser

import torch
from torch_geometric.data import DataLoader
from datasets.mixamo_skin_dataset import MixamoSkinDataset
from models.mixamo_skin_model import MixamoMeshSkinModel

def get_configs():
    current_time = datetime.datetime.now().strftime('%d%b%Y-%H:%M')
    parser = ArgumentParser(config_file_parser_class=YAMLConfigFileParser)
    parser.add_argument('-c', '--config', required=True, is_config_file=True,
                        help='config file path')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data lodaing workers', default=8)
    parser.add_argument('--nepoch', type=int, default=4, help='number of epochs to train for')
    parser.add_argument('--reproduce', default=False, action='store_true')
    parser.add_argument('--time', default=False, action='store_true')
    parser.add_argument('--datatype', default=None, type=str)  # point, point_uniform, point_poisson
    parser.add_argument('--preprocess', default=False, action='store_true')

    # Dataset Settings
    parser.add_argument('--model', type=str, default='', help='model path to load from')
    parser.add_argument('--data_dir', type=str, default='data/mixamo', help='dataset path')
    parser.add_argument('--dataset_type', type=str, default='MixamoMeshDataset', help='type of dataset')
    parser.add_argument('--version', type=str, default='', help='version of dataset')
    parser.add_argument('--split_version', type=str, default='', help='version of split')
    parser.add_argument('--no_skin', default=False, action='store_true')  # hoped it will make faster dataloader


    parser.add_argument('--log_dir', type=str, default=f"./logs/{current_time}", help='log dir')
    parser.add_argument('--save_step', type=int, default=1000, help='model saving interval')
    parser.add_argument('--vis_step', type=int, default=1000, help='model visualization interval')
    parser.add_argument('--sample_eval', type=int, default=1, help='use only subset of eval dataset')

    # Model Settings
    parser.add_argument('--npts', type=int, default=4096, dest='npts')
    parser.add_argument('--num_joints', type=int, default=22)
    parser.add_argument('--joint_loss_type', type=str, default='rel')  # 'rel'/'glob'/'rel2glob'
    parser.add_argument('--bindpose_loss_type', type=str, default='glob')  # 'rel'/'glob'/'rel2glob'
    parser.add_argument('--use_bindpose', default=False, action='store_true')
    parser.add_argument('--use_gt_ibm', default=False, action='store_true')

    parser.add_argument('--use_normal', default=False, action='store_true')
    parser.add_argument('--quantize', type=int, default=0)

    # Network Settings
    parser.add_argument('--use_bn', default=False, action='store_true', help='Use Batch Norm in networks?')
    parser.add_argument('--global_feature_size', type=int, default=1024)
    parser.add_argument('--feature_size', type=int, default=1024)
    parser.add_argument('--channels', type=int, default=[64, 256, 512], nargs=3)
    parser.add_argument('--k', type=int, default=-1)  # k for k-nearest neighbor in euclidean distance
    parser.add_argument('--euc_radius', type=float, default=0.0)  # euclidean ball, 0.6 in RigNet
    parser.add_argument('--network_type', type=str, default='full')  # k for k-nearest neighbor in euclidean distance
    parser.add_argument('--edge_type', type=str, default='tpl_and_euc', help='select one of tpl_and_euc, tpl_only, euc_only')

    # Hyperparameter Settings
    parser.add_argument('--rot_hp', type=float, default=1., help='weight of rotation loss')
    parser.add_argument('--trans_hp', type=float, default=1., help='weight of translation loss')
    parser.add_argument('--skin_hp', type=float, default=1e-3, help='weight of skin loss')
    parser.add_argument('--bm_rot_hp', type=float, default=1., help='weight of rotation loss')
    parser.add_argument('--bm_trans_hp', type=float, default=1, help='weight of translation loss')
    parser.add_argument('--bm_shape_hp', type=float, default=1e-3, help='weight of skin loss')

    # Optimization Settings
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_step_size', type=int, default=100)
    parser.add_argument('--lr_gamma', type=float, default=0.8)
    parser.add_argument('--overfit', default=False, action='store_true')
    parser.add_argument('--vis_overfit', default=False, action='store_true')  # overfit on vis dataset

    args = parser.parse_args()

    if not args.reproduce:
        manual_seed = random.randint(1, 10000)
    else:
        manual_seed = 0
    print("Random Seed: ", manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    return args


def get_dataloaders(args):
    configs = vars(args)
    version = args.version
    split_version = args.split_version
    Dataset = MixamoSkinDataset
    # if args.dataset_type == 'MixamoMeshDataset':
    #     Dataset = MixamoMeshDataset
    # elif args.dataset_type == 'MixamoPointDataset':
    # Dataset = MixamoPointDataset
    # else:
    #     raise NotImplementedError

    if args.overfit:
        # "Training done on single instance: 'aj', 'Samba Dancing_000000'
        raise NotImplementedError
        train_dataset = Dataset(data_dir=args.data_dir, split='train_overfit', version=version, split_version=split_version, preprocess=args.preprocess, datatype=args.datatype, configs=configs)
        eval_dataset = train_dataset
        vis_dataset = train_dataset
        args.batch_size = 2
    elif args.vis_overfit:
        vis_dataset = Dataset(data_dir=args.data_dir, split='test_models', version=version, split_version=split_version, preprocess=args.preprocess, datatype=args.datatype, configs=configs)
        train_dataset = vis_dataset
        eval_dataset = vis_dataset
    else:  # normal training
        train_dataset = Dataset(data_dir=args.data_dir, split='train_models', version=version, split_version=split_version, preprocess=args.preprocess, datatype=args.datatype, configs=configs)
        eval_dataset = Dataset(data_dir=args.data_dir, split='valid_models', version=version, split_version=split_version, preprocess=args.preprocess, datatype=args.datatype, configs=configs)
        vis_dataset = Dataset(data_dir=args.data_dir, split='test_models', version=version, split_version=split_version, preprocess=args.preprocess, datatype=args.datatype, configs=configs)

    if args.sample_eval > 1:
        eval_dataset = torch.utils.data.Subset(eval_dataset, list(range(0, eval_dataset.__len__(), args.sample_eval)))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers), pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers), pin_memory=True)
    vis_dataloader = DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=int(args.workers), pin_memory=True)

    return train_dataloader, eval_dataloader, vis_dataloader

def get_hyperparameters(args):
    hyper_parameters = {
        'rot_hp': args.rot_hp,
        'trans_hp': args.trans_hp,
        'skin_hp': args.skin_hp,
        'bm_rot_hp': args.bm_rot_hp,
        'bm_trans_hp': args.bm_trans_hp,
        'bm_shape_hp': args.bm_shape_hp
    }
    return hyper_parameters

def get_networkconfigs(args):
    network_configs = {
        # pointnet
        'npts': args.npts,
        'quantize': args.quantize,
        # vertex features
        'use_normal': args.use_normal,
        # euclidean edge-conv
        'euc_radius': args.euc_radius,  # euc_edge_index selected on runtime
        # edge-conv
        'global_feature_size': args.global_feature_size,
        'feature_size': args.feature_size,
        'channels': args.channels,
        # dynmic-edgeconv graph configs
        'k': args.k,
        'edge_type': args.edge_type
    }
    return network_configs