import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.nn import radius_graph

blue = lambda x: '\033[94m' + x + '\033[0m'


from models.gcn_modules import MLP, GCU
from torch_scatter import scatter_max
from torch.nn import Sequential, Dropout, Linear
import torch_geometric

class EdgeConvfeat(torch.nn.Module):
    def __init__(self, out_channels, channels=[64, 256, 512], global_channel=1024, input_normal=False, arch='all_feat',
                 aggr='max', use_bn=False, graph_configs=None):
        super(EdgeConvfeat, self).__init__()
        self.input_normal = input_normal
        self.arch = arch
        if self.input_normal:
            self.input_channel = 6 + 26 # pos(3) + norm(3) + vol_geo(26)
            raise NotImplementedError
        else:
            self.input_channel = 3 + 26 # pos(3) + vol_geo(26)

        self.global_feat_size = global_channel
        k = graph_configs['k']

        self.gcu_1 = GCU(in_channels=self.input_channel, out_channels=channels[0], k=k, aggr=aggr)
        self.gcu_2 = GCU(in_channels=channels[0], out_channels=channels[1], k=k, aggr=aggr)
        self.gcu_3 = GCU(in_channels=channels[1], out_channels=channels[2], k=k, aggr=aggr)
        # feature compression
        self.mlp_glb = MLP([(channels[0] + channels[1] + channels[2]), self.global_feat_size], use_bn=use_bn)
        if self.arch != 'global_feat':
            self.mlp_transform = Sequential(MLP([self.global_feat_size + self.input_channel + channels[0] + channels[1] + channels[2],
                                                 self.global_feat_size, 256], use_bn=use_bn),
                                            Dropout(0.7), Linear(256, out_channels))
        # edge index type
        self.edge_type = graph_configs['edge_type']

    def forward(self, mesh: torch_geometric.data.Batch):
        if self.input_normal:
             x = torch.cat([mesh.pos, mesh.x, mesh.volumetric_geodesic], dim=1)
             raise NotImplementedError
        else:
            x = torch.cat([mesh.pos, mesh.volumetric_geodesic], dim=1)

        edge_index, euc_edge_index, batch = mesh.edge_index, mesh.euc_edge_index, mesh.batch

        if self.edge_type == 'tpl_and_euc':
            x_1 = self.gcu_1(x, batch, edge_index, euc_edge_index)  # [V, channels[0]]
            x_2 = self.gcu_2(x_1, batch, edge_index, euc_edge_index)  # [V, channels[1]]
            x_3 = self.gcu_3(x_2, batch, edge_index, euc_edge_index)  # [V, channels[2]]
        elif self.edge_type == 'tpl_only':
            x_1 = self.gcu_1(x, batch, edge_index, None)  # [V, channels[0]]
            x_2 = self.gcu_2(x_1, batch, edge_index, None)  # [V, channels[1]]
            x_3 = self.gcu_3(x_2, batch, edge_index, None)  # [V, channels[2]]
        elif self.edge_type == 'euc_only':
            x_1 = self.gcu_1(x, batch, None, euc_edge_index)  # [V, channels[0]]
            x_2 = self.gcu_2(x_1, batch, None, euc_edge_index)  # [V, channels[1]]
            x_3 = self.gcu_3(x_2, batch, None, euc_edge_index)  # [V, channels[2]]
        else:
            raise NotImplementedError
        x_4 = self.mlp_glb(torch.cat([x_1, x_2, x_3], dim=1))  # [V, 1024]

        x_global_feat, _ = scatter_max(x_4, batch, dim=0)  # [B, 1024]
        if self.arch == 'global_feat':
            return x_global_feat  # [B, 1024]
        x_global = torch.repeat_interleave(x_global_feat, torch.bincount(batch), dim=0)  # global feature to each vertex
        x_5 = torch.cat([x_global, x, x_1, x_2, x_3], dim=1)  # [V, 1024+input_channel+sum(channels)]

        out = self.mlp_transform(x_5)  # [V, out_channels]
        if self.arch == 'mesh_feat':
            return out  # [V, out_channels]
        elif self.arch == 'all_feat':
            return out, x_global_feat  # [V, out_channels], [B, 1024]
        return out


class MixamoMeshSkinModel(nn.Module):
    def __init__(self, configs, num_joints=22, use_bn=False):
        super(MixamoMeshSkinModel, self).__init__()
        self.writer = SummaryWriter
        self.global_feature_size = configs['global_feature_size']
        self.feature_size = configs['feature_size']
        self.channels = configs['channels']
        self.configs = configs
        self.input_normal = configs['use_normal']
        self.edgeconv_feat = EdgeConvfeat(out_channels=self.feature_size, channels=self.channels, global_channel=self.global_feature_size,
                                          input_normal=self.input_normal, arch='all_feat', aggr='max', use_bn=use_bn, graph_configs=configs)
        self.skinnet = MeshSkinNet(input_channels=self.feature_size, num_joints=num_joints, use_bn=use_bn)

    def forward(self, data):
        # mesh = data[0]
        mesh = data
        if self.configs['euc_radius'] > 0:
            mesh.euc_edge_index = radius_graph(mesh.pos, self.configs['euc_radius'])
        else:
            mesh.euc_edge_index = None
        mesh_feat, global_feat = self.edgeconv_feat(mesh)
        skin_logits = self.skinnet(mesh_feat)
        return skin_logits

    def calculate_loss(self, pred_skin, mesh, writer=None, step=None, summary_step=1):
        """
        In:
            preds = (pred_jm_rot6d, pred_jm_trans, pred_skin) + (pred_bm_rot6d, pred_bm_trans)
            targets = (mesh, gt_jm, gt_ibm)
        Out:
            loss
        """
        self.skin_loss = 0
        l2_loss = torch.nn.MSELoss()
        batch_size = mesh.batch.max().item() + 1
        for i in range(batch_size):
            single_gt_skin, single_pred_skin = mesh.skin[mesh.batch == i], pred_skin[mesh.batch == i]
            single_skin_loss = - torch.mean(torch.sum(single_pred_skin * single_gt_skin, axis=-1))
            self.skin_loss += single_skin_loss / batch_size
        if writer is not None and self.training:
            self.write_summary(writer, step, self.skin_loss)

        return self.skin_loss

    def write_summary(self, writer, step, skin_loss=None):
        # summary is written at every step during training.
        # you have to explicitly call this function when evaluating
        if self.training:
            mode = 'train'
        else:
            mode = 'eval'
        writer.add_scalar(f'{mode}/skin_loss', skin_loss, step)

    def print_running_loss(self, epoch, i, num_batch):
        self.print_loss(epoch, i, num_batch, self.skin_loss, is_mean=False)

    def print_loss(self, epoch, i, num_batch, skin_loss=None, is_mean=False):
        mode = 'Train' if self.training else 'Eval'
        status, end = (blue(mode) + ' {}'.format(epoch), '\n') if is_mean else (mode + ' {}: {}/{}'.format(epoch, i, num_batch), '')
        print("\r[" + status + "] " +
            "skin loss: {:.4f}".format(
                skin_loss
            ), end=end)


class MeshSkinNet(nn.Module):
    def __init__(self, input_channels, num_joints, use_bn=False):
        super(MeshSkinNet, self).__init__()
        self.num_joints = num_joints
        self.use_bn = use_bn
        self.mlp_skin = MLP([input_channels, 256, 128, num_joints], use_bn)

    def forward(self, mesh_feat):
        out = self.mlp_skin(mesh_feat)  # [V, num_joints]
        log_prob = F.log_softmax(out.view(-1, self.num_joints), dim=-1)
        return log_prob
