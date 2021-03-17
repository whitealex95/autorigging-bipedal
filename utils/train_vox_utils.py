import sys
import os
import random
import datetime
import shutil
from time import time
import yaml
from configargparse import ArgumentParser, YAMLConfigFileParser

import numpy as np
import torch
# from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
from datasets.mixamo_vox_dataset import MixamoVoxDataset
blue = lambda x: '\033[94m' + x + '\033[0m'

from utils.voxel_utils import get_final_preds
from utils.joint_util import transform_rel2glob
from utils.loss_utils import normalize3d

def get_configs_from_arguments(testing=False):
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
    parser.add_argument('--data_dir', type=str, default='data/mixamo_4k', help='dataset path')
    parser.add_argument('--reduce_motion', dest='reduce_motion', action='store_true')
    parser.add_argument('--no-reduce_motion', dest='reduce_motion', action='store_false')
    parser.set_defaults(reduce_motion=True)
    # Data Setting
    parser.add_argument('--num_joints', type=int, default=22)
    parser.add_argument('--padding', type=int, default=3, help='padding of data')
    parser.add_argument('--dim_ori', type=int, default=82, help='padding of data')
    parser.add_argument('--dim_pad', type=int, default=88, help='padding of data')
    parser.add_argument('--sigma', type=float, default=2.4, help='sigma of gaussian heatmap')
    ## BVH preprocess setting
    parser.add_argument('--zero_root', default=False, action='store_true')
    parser.add_argument('--rel_rot', default=False, action='store_true')
    parser.add_argument('--augment_rot_std', type=float, default=0, help='sigma of gaussian heatmap')
    parser.add_argument('--jitter_pos_std', type=float, default=0, help='sigma of gaussian heatmap')

    # Log Settings
    parser.add_argument('--log_dir', type=str, default=f"./logs/{current_time}", help='log dir')
    parser.add_argument('--save_epoch', type=int, default=1000, help='model saving interval')
    parser.add_argument('--vis_epoch', type=int, default=1000, help='model visualization interval')
    parser.add_argument('--sample_train', type=int, default=1, help='use only subset of train dataset')


    # Network Settings
    parser.add_argument('--use_bn', default=False, action='store_true', help='Use Batch Norm in networks?')
    parser.add_argument('--network', type=str, default='HG', help='network type')
    parser.add_argument('--HG_type', type=str, default='double', help='network type')  # [double, double2: feature doubles, double2_attention]
    parser.add_argument('--feature_channels', type=int, default=24, help='network type')  # I thinkg default should be 48
    parser.add_argument('--mean_hourglass', default=False, action='store_true', help='network type')
    ## Hourglass Network Settings
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument('--n_stack', type=int, default=2)  # number of stacks
    parser.add_argument('--activation', type=str, default='none', help='activation type')
    parser.add_argument('--normalize_heatmap', default=False, action='store_true', help='normalize type')

    # Loss Settings
    parser.add_argument('--loss_type', type=str, default='mse', help='loss type')  # ['mse', 'ce', 'ce_mask']
    parser.add_argument('--loss_symmetry', type=float, default=0.0)

    # Optimization Settings
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_step_size', type=int, default=100)
    parser.add_argument('--lr_gamma', type=float, default=0.8)
    parser.add_argument('--overfit', default=False, action='store_true')
    parser.add_argument('--vis_overfit', default=False, action='store_true')  # overfit on vis dataset

    args = parser.parse_args()

    print(args)
    if not testing:
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
                    print("Retraining from specified ckpt: {}".format(args.model))
                else:
                    print("Automatically loading most recent ckpt")
                    ckpt_list = [ckpt for ckpt in os.listdir(args.log_dir) if ckpt.endswith('.pth')]
                    if len(ckpt_list) != 0:
                        args.model = os.path.join(args.log_dir,
                                                sorted(ckpt_list, key=lambda ckpt_str: ckpt_str.split('_')[-1].split('.pth')[0])[-1])
                        print("Retraining from ckpt: {}".format(args.model))


        else:
            os.makedirs(args.log_dir)
    if not testing:
        with open(os.path.join(args.log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(args), f)#,default_flow_style=None)

    if not args.reproduce:
        manual_seed = random.randint(1, 10000)
    else:
        manual_seed = 0
    print("Random Seed: ", manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    configs = vars(args)
    return configs

def train_vox(i, epoch, step, data, model, optimizer, writer, losses_dict, train_num_batch, time_log, device, configs):
    # Load data
    if configs['time'] and i < 5:
        torch.cuda.synchronize()
        time_log['after_load'] = time()
    # Send data to CUDA
    data = [dat.to(device).float() for dat in data[:-1]] + [data[-1]]  # last data is meta data
    bin_vox_padded, sdf_vox_padded, target_heatmap, JM4x4, meta = data  # unnormalized target_heatmap

    optimizer.zero_grad()
    if configs['time'] and i < 5:
        torch.cuda.synchronize()
        time_log['before_pred'] = time()
    # Single Inference
    pred_heatmap_logit = model(sdf_vox_padded)
    if configs['loss_type'].endswith('mask'):
        pred_heatmap = torch.sigmoid(pred_heatmap_logit) * bin_vox_padded
    else:
        pred_heatmap = torch.sigmoid(pred_heatmap_logit)
    if configs['time'] and i < 5:
        torch.cuda.synchronize()
        time_log['after_pred'] = time()
    # Compute Loss
    loss = joint_loss = model.compute_loss(pred_heatmap_logit, target_heatmap, bin_vox_padded)
    if configs['time'] and i < 5:
        time_log['after_loss'] = time()
    # Compute accuracy
    acc = joint_acc = model.compute_accuracy(pred_heatmap, target_heatmap, JM4x4, meta)
    losses_dict['loss'].append(loss.item())
    losses_dict['joint_acc'].append(joint_acc.item())
    if configs['time'] and i < 5:
        torch.cuda.synchronize()
        time_log['after_acc'] = time()
    # Print loss
    model.print_loss(loss, joint_acc, epoch, i, train_num_batch)
    model.write_summary(losses_dict, step=step, writer=writer)
    # Compute Gradients
    loss.backward()
    if configs['time'] and i < 5:
        torch.cuda.synchronize()
        time_log['after_backprop'] = time()
    optimizer.step()
    # Print Timing
    if configs['time'] and i < 5:
        torch.cuda.synchronize()
        time_log['after_update'] = time()
        print('\r' + blue('[Log b={:d} i={:d}]'.format(configs['batch_size'], i)) +
              ' Total: {:.3f},  Loading: {:.3f}, Inferring: {:.3f}, ComputeLoss: {:.3f}, ComputeAcc: {:.3f}, Backprop: {:.3f}, Update: {:.3f}'.format(
                  time_log['after_update'] - time_log['before_load'],
                  time_log['after_load'] - time_log['before_load'],
                  time_log['after_pred'] - time_log['before_pred'], time_log['after_loss'] - time_log['after_pred'],
                  time_log['after_acc'] - time_log['after_loss'],
                  time_log['after_backprop'] - time_log['after_acc'], time_log['after_update'] - time_log['after_backprop']
              ))
        time_log['before_load'] = time()
    # Gradient Descent

def eval_vox(i, epoch, step, data, model, writer, losses_dict, eval_num_batch, device, configs):
    data = [dat.to(device).float() for dat in data[:-1]] + [data[-1]]
    bin_vox_padded, sdf_vox_padded, target_heatmap, JM4x4, meta = data
    pred_heatmap_logit = model(sdf_vox_padded)
    if configs['loss_type'].endswith('mask'):
        pred_heatmap = torch.sigmoid(pred_heatmap_logit) * bin_vox_padded
    else:
        pred_heatmap = torch.sigmoid(pred_heatmap_logit)
    loss = joint_loss = model.compute_loss(pred_heatmap_logit, target_heatmap, bin_vox_padded)
    acc = joint_acc = model.compute_accuracy(pred_heatmap, target_heatmap, JM4x4, meta)
    losses_dict['loss'].append(loss.item())
    losses_dict['joint_acc'].append(joint_acc.item())
    model.print_loss(loss, joint_acc, epoch, i, eval_num_batch)

def vis_vox(epoch, data, model, device, configs):
    data = [dat.to(device).float() for dat in data[:-1]] + [data[-1]]
    bin_vox_padded, sdf_vox_padded, target_heatmap, JM4x4, meta = data
    if configs['normalize_heatmap']:
        target_heatmap = normalize3d(target_heatmap)
    pred_heatmap_logit = model(sdf_vox_padded)
    if configs['loss_type'].endswith('mask'):
        pred_heatmap = torch.sigmoid(pred_heatmap_logit) * bin_vox_padded
    else:
        pred_heatmap = torch.sigmoid(pred_heatmap_logit)
    acc = joint_acc = model.compute_accuracy(pred_heatmap, target_heatmap, JM4x4, meta, average=False)

    character_name, motion_name = meta['character_name'], meta['motion_name']
    batch_size = bin_vox_padded.shape[0]
    pred_coords, _ = get_final_preds(pred_heatmap, meta['translate'], meta['scale']* configs['downsample'])
    target_heatmap_coords, _ = get_final_preds(target_heatmap, meta['translate'], meta['scale'])
    JM4x4_global = transform_rel2glob(JM4x4)
    target_coords = JM4x4_global[:, :, :3, 3]

    for i in range(batch_size):
        character_name_i, motion_name_i = character_name[i], motion_name[i]
        pred_heatmap_i = pred_heatmap[i].cpu().numpy()
        target_heatmap_i = target_heatmap[i].cpu().numpy()
        pred_coords_i = pred_coords[i]
        target_coords_i = target_coords[i].cpu().numpy()
        target_heatmap_coords_i = target_heatmap_coords[i]
        acc_i = acc[i].cpu().numpy()
        if not os.path.exists(os.path.join(configs['log_dir'], 'vis', character_name_i)):
            os.makedirs(os.path.join(configs['log_dir'], 'vis', character_name_i))

        np.save(os.path.join(configs['log_dir'], 'vis', character_name_i, motion_name_i + '_pred_hm_%.3d.csv' % (epoch)),pred_heatmap_i)
        if (epoch == 0):
            np.save(os.path.join(configs['log_dir'], 'vis', character_name_i, motion_name_i + '_gt_hm_%.3d.csv' % (epoch)),target_heatmap_i)
        np.save(os.path.join(configs['log_dir'], 'vis', character_name_i, motion_name_i + '_info_%.3d.csv' % (epoch)),{
            'pred_coords': pred_coords_i,
            'target_coords': target_coords_i,
            'target_heatmap_coords': target_heatmap_coords_i,
            'acc': acc_i
        })
        # np.savetxt(os.path.join(configs['log_dir'], 'vis', character_name_i, motion_name_i + '_skin_%.3d.csv' % (epoch)),
        #            pred_skin_i, delimiter=',')

def get_vox_dataloaders(configs):
    Dataset = MixamoVoxDataset
    data_dir = configs['data_dir']
    train_split, eval_split, vis_split = 'train_models.txt', 'valid_models.txt', 'test_models.txt'
    reduce_motion = True
    if 'reduce_motion' in configs.keys() and configs['reduce_motion']==False:
        reduce_motion = False
    if configs['vis_overfit']:
        train_split = eval_split = vis_split = 'test_models.txt'
    train_dataset = Dataset(train_split, configs, reduce_motion=reduce_motion)
    eval_dataset = Dataset(eval_split, configs, reduce_motion=reduce_motion)
    vis_dataset = Dataset(vis_split, configs, reduce_motion=True)
    train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=int(configs['workers']), pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=int(configs['workers']), pin_memory=True)
    vis_dataloader = DataLoader(vis_dataset, batch_size=configs['batch_size'], shuffle=False, num_workers=int(configs['workers']), pin_memory=True)

    return train_dataloader, eval_dataloader, vis_dataloader