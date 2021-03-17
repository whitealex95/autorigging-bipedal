import torch
import torch.nn as nn
import torch.nn.functional as F

""" Code referred from https://github.com/zhan-xu/AnimSkelVolNet/blob/master/models3D/model3d_hg.py"""
class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding = ((kernel_size - 1) // 2)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)

class Pool3DBlock(nn.Module):
    def __init__(self, pool_size, input_plane):
        super(Pool3DBlock, self).__init__()
        self.stride_conv = nn.Sequential(
            nn.Conv3d(input_plane, input_plane, kernel_size=pool_size, stride=pool_size, padding=0),
            nn.BatchNorm3d(input_plane),
            nn.ReLU(True)
        )

    def forward(self, x):
        y = self.stride_conv(x)
        return y

class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, output_padding=0):
        super(Upsample3DBlock, self).__init__()
        assert (stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=output_padding),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class HG(nn.Module):
    def __init__(self, input_channels, output_channels, N=88):
        super(HG, self).__init__()
        outer_padding = [(N//4)%2, (N//2)%2, (N//1)%2]
        self.encoder_pool1 = Pool3DBlock(2, input_channels)
        self.encoder_res1 = Res3DBlock(input_channels, 16)
        self.encoder_pool2 = Pool3DBlock(2, 16)
        self.encoder_res2 = Res3DBlock(16, 24)
        self.encoder_pool3 = Pool3DBlock(2, 24)
        self.encoder_res3 = Res3DBlock(24, 36)

        self.decoder_res3 = Res3DBlock(36, 36)
        self.decoder_upsample3 = Upsample3DBlock(36, 24, 2, 2, outer_padding[0])
        self.decoder_res2 = Res3DBlock(24, 24)
        self.decoder_upsample2 = Upsample3DBlock(24, 16, 2, 2, outer_padding[1])
        self.decoder_res1 = Res3DBlock(16, 16)
        self.decoder_upsample1 = Upsample3DBlock(16, output_channels, 2, 2, outer_padding[2])

        self.skip_res1 = Res3DBlock(input_channels, output_channels)
        self.skip_res2 = Res3DBlock(16, 16)
        self.skip_res3 = Res3DBlock(24, 24)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        skip_x3 = self.skip_res3(x)
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x)

        x = self.decoder_res3(x)
        x = self.decoder_upsample3(x)
        x = x + skip_x3
        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1
        return x

class HG_double(nn.Module):
    def __init__(self, input_channels, output_channels, N=88):
        super(HG_double, self).__init__()
        outer_padding = [(N//4)%2, (N//2)%2, (N//1)%2]
        self.encoder_pool1 = Pool3DBlock(2, input_channels)
        self.encoder_res1 = Res3DBlock(input_channels, 32)
        self.encoder_pool2 = Pool3DBlock(2, 32)
        self.encoder_res2 = Res3DBlock(32, 48)
        self.encoder_pool3 = Pool3DBlock(2, 48)
        self.encoder_res3 = Res3DBlock(48, 72)

        self.decoder_res3 = Res3DBlock(72, 72)
        self.decoder_upsample3 = Upsample3DBlock(72, 48, 2, 2, outer_padding[0])
        self.decoder_res2 = Res3DBlock(48, 48)
        self.decoder_upsample2 = Upsample3DBlock(48, 32, 2, 2, outer_padding[1])
        self.decoder_res1 = Res3DBlock(32, 32)
        self.decoder_upsample1 = Upsample3DBlock(32, output_channels, 2, 2, outer_padding[2])

        self.skip_res1 = Res3DBlock(input_channels, output_channels)
        self.skip_res2 = Res3DBlock(32, 32)
        self.skip_res3 = Res3DBlock(48, 48)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        skip_x3 = self.skip_res3(x)
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x)

        x = self.decoder_res3(x)
        x = self.decoder_upsample3(x)
        x = x + skip_x3
        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1
        return x

class HG_double2(nn.Module):
    def __init__(self, input_channels, output_channels, N=88):
        super(HG_double2, self).__init__()
        outer_padding = [(N//4)%2, (N//2)%2, (N//1)%2]
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4] # , n1 * 8, n1 * 16]

        self.encoder_pool1 = Pool3DBlock(2, input_channels)
        self.encoder_res1 = Res3DBlock(input_channels, filters[0])
        self.encoder_pool2 = Pool3DBlock(2, filters[0])
        self.encoder_res2 = Res3DBlock(filters[0], filters[1])
        self.encoder_pool3 = Pool3DBlock(2, filters[1])
        self.encoder_res3 = Res3DBlock(filters[1], filters[2])

        self.decoder_res3 = Res3DBlock(filters[2], filters[2])
        self.decoder_upsample3 = Upsample3DBlock(filters[2], filters[1], 2, 2, outer_padding[0])
        self.decoder_res2 = Res3DBlock(filters[1], filters[1])
        self.decoder_upsample2 = Upsample3DBlock(filters[1], filters[0], 2, 2, outer_padding[1])
        self.decoder_res1 = Res3DBlock(filters[0], filters[0])
        self.decoder_upsample1 = Upsample3DBlock(filters[0], output_channels, 2, 2, outer_padding[2])

        self.skip_res1 = Res3DBlock(input_channels, output_channels)
        self.skip_res2 = Res3DBlock(filters[0], filters[0])
        self.skip_res3 = Res3DBlock(filters[1], filters[1])

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        skip_x3 = self.skip_res3(x)
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x)

        x = self.decoder_res3(x)
        x = self.decoder_upsample3(x)
        x = x + skip_x3
        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1
        return x

class HG_double2_attention(nn.Module):
    """ Code referred from https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py"""
    def __init__(self, input_channels, output_channels, N=88):
        super(HG_double2_attention, self).__init__()
        outer_padding = [(N//4)%2, (N//2)%2, (N//1)%2]
        n1 = output_channels * 2
        filters = [n1, n1 * 2, n1 * 4] # , n1 * 8, n1 * 16]

        self.encoder_pool1 = Pool3DBlock(2, input_channels)
        self.encoder_res1 = Res3DBlock(input_channels, filters[0])
        self.encoder_pool2 = Pool3DBlock(2, filters[0])
        self.encoder_res2 = Res3DBlock(filters[0], filters[1])
        self.encoder_pool3 = Pool3DBlock(2, filters[1])
        self.encoder_res3 = Res3DBlock(filters[1], filters[2])

        self.decoder_upsample3 = Upsample3DBlock(filters[2], filters[1], 2, 2, outer_padding[0])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.decoder_res3 = Res3DBlock(filters[2], filters[1])
        self.decoder_upsample2 = Upsample3DBlock(filters[1], filters[0], 2, 2, outer_padding[1])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=output_channels)
        self.decoder_res2 = Res3DBlock(filters[1], filters[0])
        self.decoder_upsample1 = Upsample3DBlock(filters[0], output_channels, 2, 2, outer_padding[2])

    def forward(self, x):
        e1 = self.encoder_pool1(x)
        e1 = self.encoder_res1(e1)
        e2 = self.encoder_pool2(e1)
        e2 = self.encoder_res2(e2)
        e3 = self.encoder_pool3(e2)
        e3 = self.encoder_res3(e3)

        d3 = self.decoder_upsample3(e3)
        x3 = self.Att3(g=d3, x=e2)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.decoder_res3(d3)

        d2 = self.decoder_upsample2(d3)
        x2 = self.Att2(g=d2, x=e1)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.decoder_res2(d2)

        d1 = self.decoder_upsample1(d2)
        # x1 = self.Att1(g=d1, x=x)
        # d1 = torch.cat([d1, x1])
        # d1 = self.decoder_res1(d1)

        return d1

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class V2V_HG(nn.Module):
    def __init__(self, input_channels, feature_channels, n_stack, n_joint,downsample=1, configs=None):
        super(V2V_HG, self).__init__()
        self.input_channels = input_channels
        self.n_stack = n_stack
        self.n_joint = n_joint
        self.configs = configs

        # fix feature_channels to n_joitn(=22) like in SkelVolNet
        # feature_channels = n_joint
        self.feature_channels = feature_channels
        if configs['HG_type'] == 'double':
            HG = HG_double
        elif configs['HG_type'] == 'double2':
            HG = HG_double
        elif configs['HG_type'] == 'double2_attention':
            HG = HG_double2_attention

        if downsample>1:
            self.front_layers = nn.Sequential(
                Basic3DBlock(input_channels, feature_channels, 5),
                Res3DBlock(feature_channels, feature_channels),
                Pool3DBlock(int(downsample), feature_channels)
            )
        else:
            self.front_layers = nn.Sequential(
                Basic3DBlock(input_channels, feature_channels, 5),
                Res3DBlock(feature_channels, feature_channels)
            )
        self.hg_1 = HG(input_channels=feature_channels, output_channels=feature_channels, N=88//downsample)
        self.joint_output_1 = nn.Sequential(
            Res3DBlock(feature_channels, feature_channels//2),
            Basic3DBlock(feature_channels//2, feature_channels//2, 1),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(feature_channels//2, n_joint, kernel_size=1, stride=1, padding=0)
        )

        if n_stack > 1:
            self.hg_list = nn.ModuleList([HG(input_channels=feature_channels+n_joint, output_channels=feature_channels, N=88//downsample) for i in range(1, n_stack)])
            self.joint_output_list = nn.ModuleList([nn.Sequential(
                Res3DBlock(feature_channels, feature_channels//2), Basic3DBlock(feature_channels//2, feature_channels//2, 1), nn.Dropout3d(p=0.2),
                nn.Conv3d(feature_channels//2, n_joint, kernel_size=1, stride=1, padding=0)) for i in range(1, n_stack)])
        self._initialize_weights()

    def forward(self, x_in):
        x = x_in.unsqueeze(dim=1)
        x = self.front_layers(x)
        x_hg_1 = self.hg_1(x)
        x_joint = self.joint_output_1(x_hg_1)
        x_joint_out = [x_joint]

        for i in range(1, self.n_stack):
            x_in = torch.cat((x, x_joint), dim=1)
            x_hg = self.hg_list[i-1](x_in)
            x_joint = self.joint_output_list[i-1](x_hg)
            x_joint_out.append(x_joint)

        if self.configs['mean_hourglass'] and self.train():
            for x_joint_single in x_joint_out[:-1]:
                x_joint += x_joint_single
            x_joint = torch.true_divide(x_joint, len(x_joint_out))
        return x_joint

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


