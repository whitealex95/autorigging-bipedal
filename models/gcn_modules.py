import torch
from torch_geometric.nn import MessagePassing, knn_graph
from torch_scatter import scatter_max, scatter_mean
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch.nn import Sequential, Dropout, Linear, ReLU, BatchNorm1d, Parameter

def MLP(channels, use_bn=True):
    if use_bn:
        return Sequential(*[Sequential(Linear(channels[i - 1], channels[i]), ReLU(), BatchNorm1d(channels[i], momentum=0.1))
                            for i in range(1, len(channels))])
    else:
        return Sequential(*[Sequential(Linear(channels[i - 1], channels[i]), ReLU()) for i in range(1, len(channels))])


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, nn, aggr='max', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn

    def forward(self, x, edge_index):
        x = x.unsqueeze(-1) if x.dim() == 1 else x ## ToDo: verify
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
        return self.propagate(edge_index, x=x)  # x: [V, out_channels]

    def message(self, x_i, x_j):
        return self.nn(torch.cat([x_i, (x_j - x_i)], dim=1))

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.out_channels)
        return aggr_out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, nn, k=20, aggr='max', **kwargs):
        super(DynamicEdgeConv, self).__init__(in_channels, out_channels, nn, aggr, **kwargs)
        self.k = k

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super(DynamicEdgeConv, self).forward(x, edge_index)

class GCU(torch.nn.Module):  # graph convolution unit : Mesh -> feature vector
    def __init__(self, in_channels, out_channels, k=-1, aggr='max'):
        super(GCU, self).__init__()
        assert out_channels % 2 == 0
        # Edge conv based on mesh
        self.k = k
        self.edge_conv_0 = EdgeConv(in_channels=in_channels, out_channels=out_channels//2,
                                      nn=MLP([in_channels * 2, out_channels // 2, out_channels // 2]), aggr=aggr)
        # Dynamic Edge conv based on volumetric euclidean distance
        if k > 0:
            self.edge_conv_1 = DynamicEdgeConv(in_channels=in_channels, out_channels=out_channels//2,
                                          nn=MLP([in_channels * 2, out_channels // 2, out_channels // 2]), k=k, aggr=aggr)
        else:
            self.edge_conv_1 = EdgeConv(in_channels=in_channels, out_channels=out_channels//2,
                                          nn=MLP([in_channels * 2, out_channels // 2, out_channels // 2]), aggr=aggr)
        self.mlp = MLP([out_channels, out_channels])

    def forward(self, x, batch=None, tpl_edge_index=None, euc_edge_index=None, radius=0.):
        # assuming that at least one of (tpl_edge_index or euc_edge_index) is not None.
        if tpl_edge_index is not None:
            x0 = self.edge_conv_0(x, tpl_edge_index)  # [V, out_channels//2]
        else:
            x0 = self.edge_conv_0(x, euc_edge_index)
        if self.k > 0:
            x1 = self.edge_conv_1(x, batch)  # [V, out_channels//2]
        else:
            if euc_edge_index is not None:
                x1 = self.edge_conv_1(x, euc_edge_index)
            else:
                x1 = self.edge_conv_1(x, tpl_edge_index)
        # x_euc = self.edge_conv_euc(x, batch)  # [V, out_channels//2]
        x_out = torch.cat([x0, x1], dim=1)  # [V, out_channels]
        x_out = self.mlp(x_out)  # [V, out_channels]
        return x_out
