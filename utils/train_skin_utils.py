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
    parser.add_argument('--no_skin', default=False, action='store_true')  # hoped it will make faster dataloader


    parser.add_argument('--log_dir', type=str, default=f"./logs/{current_time}", help='log dir')
    parser.add_argument('--save_step', type=int, default=1000, help='model saving interval')
    parser.add_argument('--vis_step', type=int, default=1000, help='model visualization interval')

    # Model Settings
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

    print(args)
    if os.path.exists(args.log_dir):
        print("\nAre you re-training? [y/n]", end='')
        choice = input().lower()
        if choice not in ['y', 'n']:
            print("please type in valid response")
            sys.exit()
        elif choice == 'n':
            print("The log directory is already occupied. Do you want to remove and rewrite? [y/n]", end='')
            choice = input().lower()
            if choice == 'y':
                shutil.rmtree(args.log_dir, ignore_errors=True)
                os.makedirs(args.log_dir)
            else:
                print("Please choose a different log_dir")
                sys.exit()
        else:
            if args.model != '':
                print("You cannot restart when the model is specified")
                __import__('pdb').set_trace()
            else:
                ckpt_list = [ckpt for ckpt in os.listdir(args.log_dir) if ckpt.endswith('.pth')]
                args.model = os.path.join(args.log_dir,
                                          sorted(ckpt_list, key=lambda ckpt_str: ckpt_str.split('_')[-1].split('.pth')[0])[-1])

            print("Retraining from ckpt: {}".format(args.model))

    else:
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)#, default_flow_style=None)

    if not args.reproduce:
        manual_seed = random.randint(1, 10000)
    else:
        manual_seed = 0
    print("Random Seed: ", manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    return args


def get_skin_dataloaders(args):
    configs = vars(args)
    Dataset = MixamoSkinDataset

    if args.overfit:
        # "Training done on single instance: 'aj', 'Samba Dancing_000000'
        raise NotImplementedError
        train_dataset = Dataset(data_dir=args.data_dir, split='train_overfit',preprocess=args.preprocess, datatype=args.datatype, configs=configs)
        eval_dataset = train_dataset
        vis_dataset = train_dataset
        args.batch_size = 2
    elif args.vis_overfit:
        vis_dataset = Dataset(data_dir=args.data_dir, split='test_models', preprocess=args.preprocess, datatype=args.datatype, configs=configs)
        train_dataset = vis_dataset
        eval_dataset = vis_dataset
    else:  # normal training
        train_dataset = Dataset(data_dir=args.data_dir, split='train_models', preprocess=args.preprocess, datatype=args.datatype, configs=configs)
        eval_dataset = Dataset(data_dir=args.data_dir, split='valid_models', preprocess=args.preprocess, datatype=args.datatype, configs=configs)
        vis_dataset = Dataset(data_dir=args.data_dir, split='test_models', preprocess=args.preprocess, datatype=args.datatype, configs=configs)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers), pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers), pin_memory=True)
    vis_dataloader = DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=int(args.workers), pin_memory=True)

    return train_dataloader, eval_dataloader, vis_dataloader
