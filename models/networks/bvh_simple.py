import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBVHNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(SimpleBVHNet, self).__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.fc_list = []
        self.fc_list += [nn.Linear(in_dim, hidden_dim[0])]
        self.fc_list += [nn.Linear(hidden_dim[i], hidden_dim[i+1]) for i in range(len(hidden_dim)-1)]
        self.fc_list += [nn.Linear(hidden_dim[-1], out_dim)]
        self.fc_list = nn.ModuleList(self.fc_list)
        self.activation = nn.ReLU()


    def forward(self, x):
        """
        Input: input_pos [B, J, in_dim//J]
        Output: rotation_param [B, J, out_dim//J]
        """
        x = x.reshape([-1, self.in_dim])  # [B, J, in_dim//J] -> [B, J]
        for fc in self.fc_list[:-1]:
            x = self.activation(fc(x))
        x = self.fc_list[-1](x)
        return x.view(-1, 22, self.out_dim//22)  # [B, out_dim] -> [B, J, out_dim//J]

class SimpleBVHNet_BN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(SimpleBVHNet_BN, self).__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.fc_list = []
        self.fc_list += [nn.Linear(in_dim, hidden_dim[0])]
        self.fc_list += [nn.Linear(hidden_dim[i], hidden_dim[i+1]) for i in range(len(hidden_dim)-1)]
        self.fc_list += [nn.Linear(hidden_dim[-1], out_dim)]
        self.fc_list = nn.ModuleList(self.fc_list)

        # self.bn_list = [nn.BatchNorm1d(dim) for dim in hidden_dim[:-1]]
        self.bn_list = [nn.BatchNorm1d(dim) for dim in hidden_dim]
        self.bn_list = nn.ModuleList(self.bn_list)
        self.activation = nn.ReLU()


    def forward(self, x):
        """
        Input: input_pos [B, J, in_dim//J]
        Output: rotation_param [B, J, out_dim//J]
        """
        x = x.reshape([-1, self.in_dim])  # [B, J, in_dim//J] -> [B, J]
        for layer_idx, fc in enumerate(self.fc_list[:-1]):
            x = fc(x)
            x = self.activation(x)
            x = self.bn_list[layer_idx](x)
        x = self.fc_list[-1](x)
        return x.view(-1, 22, self.out_dim//22)  # [B, out_dim] -> [B, J, out_dim//J]
