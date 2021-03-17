import os
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import defaultdict

from utils.joint_util import save_jm, save_jm2
from utils.train_skin_utils import get_skin_dataloaders, get_configs
from models.mixamo_skin_model import MixamoMeshSkinModel


args = get_configs()
configs = vars(args)  # class fields to dictionary (args.lr -> configs['lr'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dataloader, eval_dataloader, vis_dataloader = get_skin_dataloaders(args)
train_num_batch, eval_num_batch = len(train_dataloader), len(eval_dataloader)
model = MixamoMeshSkinModel(configs, num_joints=configs['num_joints'], use_bn=configs['use_bn'])
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
writer = SummaryWriter(log_dir=args.log_dir)

step = 0
if args.model != '':  # Load model
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    step = checkpoint['last_step'] + 1
    print("[Info] Loaded model parameters from " + args.model)


for epoch in range(scheduler.last_epoch, args.nepoch):
    # total_loss_list, rot_loss_list, trans_loss_list, skin_loss_list = [], [], [], []
    losses_list = []
    for i, data in enumerate(train_dataloader, 0):
        data = [dat.to(device) for dat in data[:-2]] + data[-2:]
        mesh, gt_jm, gt_ibm, character_name, motion_name = data  # jm: relative, ibm: global
        optimizer.zero_grad()
        model.train()
        pred_skin_logit = model(mesh)
        skin_loss = model.calculate_loss(pred_skin_logit, mesh, writer=writer, step=step)
        model.print_running_loss(epoch, i, train_num_batch)
        losses_list.append([skin_loss.item()])
        skin_loss.backward()
        optimizer.step()
        step += 1

    losses_mean = np.mean(losses_list)
    model.print_loss(epoch, i, train_num_batch, skin_loss=losses_mean, is_mean=True)
    model.write_summary(writer=writer, step=step, skin_loss=losses_mean)
    scheduler.step()  # schedule based on epochs

    # Evaluate
    model.eval()
    with torch.no_grad():
        losses_list = []
        for i, data in enumerate(eval_dataloader, 0):
            data = [dat.to(device) for dat in data[:-2]] + data[-2:]
            mesh, gt_jm, gt_ibm, character_name, motion_name = data  # jm: relative, ibm: global
            pred_skin_logit = model(mesh)
            skin_loss = model.calculate_loss(pred_skin_logit, mesh, writer=writer, step=step)
            model.print_running_loss(epoch, i, eval_num_batch)
            losses_list.append([skin_loss.item()])

        losses_mean = np.mean(losses_list)
        model.print_loss(epoch, i, eval_num_batch, skin_loss=losses_mean, is_mean=True)
        model.write_summary(writer, step, skin_loss=losses_mean)

    # Save training model
    if epoch % args.save_step == 0:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'last_step': step-1},
                   '%s/model_epoch_%.3d.pth' % (args.log_dir, epoch))

    # Save inference result of training model
    model.eval()
    with torch.no_grad():
        if epoch % args.vis_step == 0:
            print("Visualizing")
            for data in vis_dataloader:
                data = [dat.to(device) for dat in data[:-2]] + list(data[-2:])
                mesh, gt_jm, gt_ibm, character_name, motion_name = data
                pred_skin_logit = model(mesh)

                batch_size = mesh.batch.max().item() + 1
                for i in range(batch_size):
                    pred_skin = torch.exp(pred_skin_logit)[mesh.batch==i].cpu().detach().numpy()
                    character_name, motion_name = character_name[i], motion_name[i]

                    if not os.path.exists(os.path.join(args.log_dir, 'vis', character_name)):
                        os.makedirs(os.path.join(args.log_dir, 'vis', character_name))

                    np.savetxt(os.path.join(args.log_dir, 'vis', character_name, motion_name + '_skin_%.3d.csv' % (epoch)),
                               pred_skin, delimiter=',')