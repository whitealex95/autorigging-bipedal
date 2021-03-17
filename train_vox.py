import os
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import defaultdict

from utils.joint_util import save_jm, save_jm2
from utils.train_vox_utils import get_vox_dataloaders, train_vox, eval_vox, vis_vox, get_configs_from_arguments
from models.mixamo_vox_model import MixamoVoxModel
from time import time

configs = get_configs_from_arguments()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataloader, eval_dataloader, vis_dataloader = get_vox_dataloaders(configs)
train_num_batch, eval_num_batch = len(train_dataloader), len(eval_dataloader)
model = MixamoVoxModel(configs)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=configs['lr'], betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs['lr_step_size'], gamma=configs['lr_gamma'])
writer = SummaryWriter(log_dir=configs['log_dir'])

step = 0
start_epoch = 0
if configs['model'] != '':  # Load model
    checkpoint = torch.load(configs['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    step = checkpoint['last_step'] + 1
    start_epoch = scheduler.last_epoch
    print("[Info] Loaded model parameters from " + configs['model'])

time_log = defaultdict(float)

for epoch in range(start_epoch, start_epoch + configs['nepoch']):
    # total_loss_list, rot_loss_list, trans_loss_list, skin_loss_list = [], [], [], []
    model.train()
    losses_dict = defaultdict(list)
    losses_mean = defaultdict(float)
    configs['time'] = configs['time'] and epoch == start_epoch
    if configs['time']:
        time_log['before_load'] = time()
    for i, data in enumerate(train_dataloader, 0):
        train_vox(i, epoch, step, data, model, optimizer, writer, losses_dict, train_num_batch, time_log, device,
                  configs)
        step += 1
    for key, val in losses_dict.items():
        losses_mean[key] = np.mean(val)
    model.print_loss(losses_mean['loss'], losses_mean['joint_acc'], epoch, 0, 0); print()
    model.write_summary(losses_mean, epoch=epoch, writer=writer)
    scheduler.step()  # schedule based on epochs

    # Evaluate
    if not configs['vis_overfit']:
        model.eval()
        with torch.no_grad():
            losses_dict = defaultdict(list)
            losses_mean = defaultdict(float)
            for i, data in enumerate(eval_dataloader, 0):
                eval_vox(i, epoch, step, data, model, writer, losses_dict, train_num_batch, device, configs)
            for key, val in losses_dict.items():
                losses_mean[key] = np.mean(val)
            model.print_loss(losses_mean['loss'], losses_mean['joint_acc'], epoch, 0, 0);print()
            model.write_summary(losses_mean, epoch=epoch, writer=writer)

    # Save training model
    if epoch % configs['save_epoch'] == 0:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'last_step': step-1},
                   '%s/model_epoch_%.3d.pth' % (configs['log_dir'], epoch))

    model.eval()
    with torch.no_grad():
        if epoch % configs['vis_epoch'] == 0:
            print("Visualizing")
            for data in vis_dataloader:
                vis_vox(epoch, data, model, device, configs)

