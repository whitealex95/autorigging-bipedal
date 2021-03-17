import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndimage

import utils.binvox_rw as binvox_rw
import math

def bin2sdf(binvox):
    """
    convert binary voxels into sign distance function field. Negetive for interior. Positive for exterior. Normalized.
    :param binvox: binary voxels
    :return: SDF representation of voxel.
    TODO: This is not 'correct' bin2sdf.
        - all sdfvox are normalized differently!
            - Try finding out the maximum value for correct normalization
        - truncated representation!
        - same normalization for inside/outside voxel
    """
    fill_map = np.zeros(binvox.shape, dtype=np.bool)
    sdfvox = np.zeros(binvox.shape, dtype=np.float16)
    # fill inside
    changing_map = binvox.copy()
    sdf_in = -1
    while np.sum(fill_map) != np.sum(binvox):
        changing_map_new = ndimage.binary_erosion(changing_map)
        fill_map[np.where(changing_map_new!=changing_map)] = True
        sdfvox[np.where(changing_map_new!=changing_map)] = sdf_in
        changing_map = changing_map_new.copy()
        sdf_in -= 1
    # fill outside. No need to fill all of them, since during training, outside part will be masked.
    changing_map = binvox.copy()
    sdf_out = 1
    while np.sum(fill_map) != np.size(binvox):
        changing_map_new = ndimage.binary_dilation(changing_map)
        fill_map[np.where(changing_map_new!=changing_map)] = True
        sdfvox[np.where(changing_map_new!=changing_map)] = sdf_out
        changing_map = changing_map_new.copy()
        sdf_out += 1
        if sdf_out == -sdf_in:
            break
    # Normalization
    sdfvox[np.where(sdfvox < 0)] /= (-sdf_in-1)
    sdfvox[np.where(sdfvox > 0)] /= (sdf_out-1)
    return sdfvox


def Cartesian2Voxcoord(v, translate, scale, resolution=82):
    vc = (v - translate) / scale * resolution
    vc = np.round(vc).astype(int)
    return vc[0], vc[1], vc[2]


def Voxcoord2Cartesian(vc, translate, scale, resolution=82):
    v = vc / resolution * scale + translate
    return v[0], v[1], v[2]

def Voxbatch2Cartesian(vox_batch, translate, scale, resolution=82):
    if isinstance(vox_batch, torch.Tensor):
        v = torch.true_divide(vox_batch, resolution) * scale + translate
        return v
    v = vox_batch / resolution * scale + translate
    return v


def draw_jointmap(img, pt, sigma):
    """
    Input:
        img: 3D image
        pt: center of gaussian
        sigma: variance of gaussian (scalar)
    Output:
        3D Image of img.shape with unnormalized gaussian centered at pt
    """
    # Draw a 3D gaussian
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma), int(pt[2] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1), int(pt[2] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[0] or ul[1] >= img.shape[1] or ul[2] >= img.shape[2] or
            br[0] < 0 or br[1] < 0 or br[2] < 0):
        # if not, return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)  # [0., 1., ..., size-1] (size,)
    y = x[:, np.newaxis]  # (size, 1)
    z = y[..., np.newaxis]  # (size, 1, 1)
    x0 = y0 = z0 = size // 2
    # Gaussian is not normalized with center value equal to 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2 * sigma ** 2))  # (size, size, size)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[0]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[1]) - ul[1]
    g_z = max(0, -ul[2]), min(br[2], img.shape[2]) - ul[2]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[0])
    img_y = max(0, ul[1]), min(br[1], img.shape[1])
    img_z = max(0, ul[2]), min(br[2], img.shape[2])

    img[img_x[0]:img_x[1], img_y[0]:img_y[1], img_z[0]:img_z[1]] = \
        np.maximum(img[img_x[0]:img_x[1], img_y[0]:img_y[1], img_z[0]:img_z[1]],
                   g[g_x[0]:g_x[1], g_y[0]:g_y[1], g_z[0]:g_z[1]])
    return img

def draw_bonemap(heatmap, p_pos, c_pos, output_resulotion):
    # create 3D bone heatmap. Voxels along the bone have value 1, otherwise 0
    c_pos = np.asarray(c_pos)
    ray = c_pos - p_pos
    i_step = np.arange(1, 100)
    unit_step = (ray / 100)[np.newaxis,:]
    unit_step = np.repeat(unit_step, 99, axis=0)
    pos_step = p_pos + unit_step * i_step[:,np.newaxis]
    pos_step = np.round(pos_step).astype(np.uint8)
    pos_step = np.array([p for p in pos_step if np.all(p >= 0) and np.all(p < output_resulotion)])
    if len(pos_step) != 0:
        heatmap[pos_step[:, 0], pos_step[:, 1], pos_step[:, 2]] += 1
    np.clip(heatmap, 0.0, 1.0, out=heatmap)
    return heatmap

def center_vox(volumn_input):
    #put the occupied voxels at the center instead of corner
    pos = np.where(volumn_input > 0)
    x_max, x_min = np.max(pos[0]), np.min(pos[0])
    y_max, y_min = np.max(pos[1]), np.min(pos[1])
    z_max, z_min = np.max(pos[2]), np.min(pos[2])
    side_length = volumn_input.shape[0]
    mid_len = int(side_length / 2)
    xr_low = int((x_max - x_min + 1) / 2)
    xr_high = x_max - x_min + 1 - xr_low
    yr_low = int((y_max - y_min + 1) / 2)
    yr_high = y_max - y_min + 1 - yr_low
    zr_low = int((z_max - z_min + 1) / 2)
    zr_high = z_max - z_min + 1 - zr_low
    content = volumn_input[x_min: x_max + 1, y_min: y_max + 1, z_min: z_max + 1]
    volumn_output = np.zeros((volumn_input.shape), dtype=np.bool)
    center_trans = [x_min - mid_len + xr_low, y_min - mid_len + yr_low, z_min - mid_len + zr_low]
    center_trans = list(map(int, center_trans))
    volumn_output[mid_len - xr_low:mid_len + xr_high, mid_len - yr_low:mid_len + yr_high,
    mid_len - zr_low:mid_len + zr_high] = content
    return volumn_output, center_trans

def get_max_preds_torch(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: np.ndarray([batch_size, num_joints, dim_x, dim_y, dim_z])
    """
    assert batch_heatmaps.ndim == 5, 'batch_images should be 5-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    dim_pad = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    maxvals, idx = torch.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    preds = idx.reshape((batch_size, num_joints, 1)).repeat(1, 1, 3)
    preds[:, :, 0] = torch.floor_divide((preds[:, :, 0]), (dim_pad * dim_pad))
    preds[:, :, 1] = torch.floor_divide(preds[:, :, 1] % (dim_pad * dim_pad), dim_pad)
    preds[:, :, 2] = (preds[:, :, 2]) % dim_pad

    return preds, maxvals
def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: np.ndarray([batch_size, num_joints, dim_x, dim_y, dim_z])
    """
    if not isinstance(batch_heatmaps, np.ndarray): # 'batch_heatmaps should be numpy.ndarray'
        # return get_max_preds_torch(batch_heatmaps)
        batch_heatmaps = batch_heatmaps.cpu().detach().numpy()
    assert batch_heatmaps.ndim == 5, 'batch_images should be 5-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    dim_pad = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 3)).astype(np.float32)

    preds[:, :, 0] = np.floor((preds[:, :, 0]) / (dim_pad * dim_pad))
    preds[:, :, 1] = np.floor((preds[:, :, 1]) % (dim_pad * dim_pad) / dim_pad)
    preds[:, :, 2] = (preds[:, :, 2]) % dim_pad

    return preds, maxvals
def soft_argmax(voxels, alpha=1.0):
    """
    Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
    alpha: inverse temperature (1.0 for normal, >>1 for max)
    Return: 3D coordinates in shape (batch_size, channel, 3)
    """
    if isinstance(voxels, np.ndarray):
        conversion = True
        voxels = torch.from_numpy(voxels)
    else:
        conversion = False
    assert voxels.dim()==5
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    N,C,H,W,D = voxels.shape
    soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
    # soft_max = voxels.view(N,C,-1) / voxels.view(N,C,-1).sum(dim=-1, keepdim=True)# nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
    soft_max = soft_max.view(voxels.shape)
    indices_kernel = torch.arange(start=0,end=H*W*D).unsqueeze(0).to(voxels.device)
    x_indices_kernel = (indices_kernel.view((H,W,D))//D//W%H).clone()
    y_indices_kernel = (indices_kernel.view((H,W,D))//D%W).clone()
    z_indices_kernel = (indices_kernel.view((H,W,D))%D).clone()
    x_conv = soft_max*x_indices_kernel
    y_conv = soft_max*y_indices_kernel
    z_conv = soft_max*z_indices_kernel
    x_indices = x_conv.sum(2).sum(2).sum(2)
    y_indices = y_conv.sum(2).sum(2).sum(2)
    z_indices = z_conv.sum(2).sum(2).sum(2)
    coords = torch.stack([x_indices,y_indices,z_indices],dim=2)
    if conversion:
        coords = np.array(coords)
    return coords

def get_softmax_preds(batch_heatmaps):
    return soft_argmax(batch_heatmaps)

def get_final_preds(batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)
    center, scale = torch.stack(center, axis=1).cpu().detach().numpy()[:,None,:], np.array(scale)[:,None,None]
    return Voxbatch2Cartesian(coords, center, scale, resolution=82), maxvals

def get_final_preds_torch(batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds_torch(batch_heatmaps)
    return Voxbatch2Cartesian(coords, center, scale, resolution=82), maxvals



    # heatmap_height = batch_heatmaps.shape[2]
    # heatmap_width = batch_heatmaps.shape[3]
    #
    # # post-processing
    # if config.TEST.POST_PROCESS:
    #     for n in range(coords.shape[0]):
    #         for p in range(coords.shape[1]):
    #             hm = batch_heatmaps[n][p]
    #             px = int(math.floor(coords[n][p][0] + 0.5))
    #             py = int(math.floor(coords[n][p][1] + 0.5))
    #             if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
    #                 diff = np.array(
    #                     [
    #                         hm[py][px+1] - hm[py][px-1],
    #                         hm[py+1][px]-hm[py-1][px]
    #                     ]
    #                 )
    #                 coords[n][p] += np.sign(diff) * .25
    #
    # preds = coords.copy()
    #
    # # Transform back
    # for i in range(coords.shape[0]):
    #     preds[i] = transform_preds(
    #         coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
    #     )
    #
    # return preds, maxvals

def downsample_heatmap(batch_heatmaps, downsample):
    assert isinstance(downsample, int)
    assert batch_heatmaps.dim() == 5
    # B, J, N = batch_heatmaps.shape[0], batch_heatmaps.shape[1], batch_heatmaps.shape[2]
    # downsampled_heatmaps = torch.zeros(B, J, N//downsample, N//downsample, N//downsample)
    return F.interpolate(batch_heatmaps, scale_factor=1./downsample)

def downsample_single_heatmap(heatmap, downsample):
    assert isinstance(downsample, int)
    if isinstance(heatmap, np.ndarray):
        revert = True
        heatmap = torch.from_numpy(heatmap)
    else:
        revert = False
    # B, J, N = batch_heatmaps.shape[0], batch_heatmaps.shape[1], batch_heatmaps.shape[2]
    # downsampled_heatmaps = torch.zeros(B, J, N//downsample, N//downsample, N//downsample)
    out = F.interpolate(heatmap[None, ...], scale_factor=1./downsample)[0]
    if revert:
        out = out.cpu().detach().numpy()
    return out

def upsample_heatmap(batch_heatmaps, upsample):
    assert isinstance(upsample, int)
    assert batch_heatmaps.dim() == 5
    return F.interpolate(batch_heatmaps, scale_factor=upsample)


def extract_joint_pos_from_heatmap_softargmax(input_heatmap, scale, translate, use_downsample, center_trans, mask=None):
    if mask is not None:
        heatmap = input_heatmap.copy() * mask
    else:
        heatmap = input_heatmap
    pos = soft_argmax(heatmap[None, ...])[0]
    if use_downsample:
        pos = pos * 2 + np.array(center_trans) - 3 + 1 #+ 0.5
    else:
        pos = pos + np.array(center_trans) - 3 + 1
    pos = pos * scale / 82 + np.array(translate)
    return pos


def extract_joint_pos_from_heatmap(input_heatmap, scale, translate, use_downsample, center_trans, mask=None):
    if mask is not None:
        heatmap = input_heatmap.copy() * mask
    else:
        heatmap = input_heatmap
    pos = get_max_preds(heatmap[None, ...])[0][0]
#     print(heatmap.shape, mask.shape if mask is not None else mask, pos.shape)
    if use_downsample:
        pos = pos * 2 + np.array(center_trans) - 3 + 1 #+ 0.5
    else:
        pos = pos + np.array(center_trans) - 3 + 1
    pos = pos * scale / 82 + np.array(translate)
    return pos
