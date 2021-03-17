import sys
import os
import random
import datetime
import shutil
import yaml
from configargparse import ArgumentParser
from time import time

import numpy as np
import torch
# from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
from datasets.mixamo_bvh_dataset import MixamoBVHDataset
blue = lambda x: '\033[94m' + x + '\033[0m'

from utils.voxel_utils import get_final_preds
from utils.joint_util import transform_rel2glob
from utils.loss_utils import normalize3d

def train_bvh(i, epoch, step, data, model, optimizer, writer, losses_dict, train_num_batch, time_log, device, configs):
    # Load data
    if configs['time'] and i < 5:
        torch.cuda.synchronize()
        time_log['after_load'] = time()
    # Send data to CUDA
    data = [dat.to(device).float() for dat in data[:-1]] + [data[-1]]  # last data is meta data
    input_position, target_rotation, meta = data  #
    for key in meta:
        if isinstance(meta[key], torch.Tensor):
            meta[key].to(device).float()

    optimizer.zero_grad()
    if configs['time'] and i < 5:
        torch.cuda.synchronize()
        time_log['before_pred'] = time()
    # Single Inference
    pred_rotation = model(input_position)
    if configs['time'] and i < 5:
        torch.cuda.synchronize()
        time_log['after_pred'] = time()
    # Compute Loss
    loss =joint_loss = model.compute_loss(pred_rotation, target_rotation)
    if configs['time'] and i < 5:
        time_log['after_loss'] = time()
    # # Compute accuracy
    acc = joint_acc = model.compute_accuracy(input_position, pred_rotation, target_rotation)
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

def eval_bvh(i, epoch, step, data, model, writer, losses_dict, eval_num_batch, device, configs):
    data = [dat.to(device).float() for dat in data[:-1]] + [data[-1]]  # last data is meta data
    input_position, target_rotation, meta = data  #
    for key in meta:
        if isinstance(meta[key], torch.Tensor):
            meta[key].to(device).float()
    pred_rotation = model(input_position)
    loss =joint_loss = model.compute_loss(pred_rotation, target_rotation)
    acc = joint_acc = model.compute_accuracy(input_position, pred_rotation, target_rotation)
    losses_dict['loss'].append(loss.item())
    losses_dict['joint_acc'].append(joint_acc.item())
    model.print_loss(loss, joint_acc, epoch, i, eval_num_batch)

def vis_bvh(epoch, data, model, device, configs):
    data = [dat.to(device).float() for dat in data[:-1]] + [data[-1]]
    input_position, target_rotation, meta = data  #
    for key in meta:
        if isinstance(meta[key], torch.Tensor):
            meta[key].to(device).float()
    pred_rotation = model(input_position)
    loss =joint_loss = model.compute_loss(pred_rotation, target_rotation, average=False)
    acc = joint_acc = model.compute_accuracy(input_position, pred_rotation, target_rotation, average=False)

    character_name, motion_name = meta['character_name'], meta['motion_name']
    batch_size = input_position.shape[0]
    for i in range(batch_size):
        character_name_i, motion_name_i = character_name[i], motion_name[i]
        if not os.path.exists(os.path.join(configs['log_dir'], 'vis', character_name_i)):
            os.makedirs(os.path.join(configs['log_dir'], 'vis', character_name_i))

        np.save(os.path.join(configs['log_dir'], 'vis', character_name_i, motion_name_i + '_input_pos_%.3d.csv' % (epoch)),input_position[i].cpu().detach())
        np.save(os.path.join(configs['log_dir'], 'vis', character_name_i, motion_name_i + '_pred_rot_%.3d.csv' % (epoch)),pred_rotation[i].cpu().detach())
        np.save(os.path.join(configs['log_dir'], 'vis', character_name_i, motion_name_i + '_target_rot_%.3d.csv' % (epoch)),target_rotation[i].cpu().detach())
        np.save(os.path.join(configs['log_dir'], 'vis', character_name_i, motion_name_i + '_info_%.3d.csv' % (epoch)),{
            'loss': loss[i].cpu().detach(),
            'acc': acc[i].cpu().detach()
        })
        # np.savetxt(os.path.join(configs['log_dir'], 'vis', character_name_i, motion_name_i + '_skin_%.3d.csv' % (epoch)),
        #            pred_skin_i, delimiter=',')

def get_bvh_dataloaders(configs):
    Dataset = MixamoBVHDataset
    data_dir = configs['data_dir']
    train_split, eval_split, vis_split = 'train_models.txt', 'valid_models.txt', 'test_models.txt'
    if configs['vis_overfit']:
        train_split = eval_split = vis_split = 'test_models.txt'
    train_dataset = Dataset(train_split, configs)
    eval_dataset = Dataset(eval_split, configs)
    vis_dataset = Dataset(vis_split, configs)
    train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=int(configs['workers']), pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=int(configs['workers']), pin_memory=True)
    vis_dataloader = DataLoader(vis_dataset, batch_size=configs['batch_size'], shuffle=False, num_workers=int(configs['workers']), pin_memory=True)

    return train_dataloader, eval_dataloader, vis_dataloader

