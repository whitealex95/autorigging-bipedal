import numpy as np
import binvox_rw as binvox
import os

def read_binvox(filename):
    with open(filename, 'rb') as f:
        model = binvox.read_as_3d_array(f)
        return model.data

def save_binvox(filename, data):
    dims = data.shape
    translate = [0.0, 0.0, 0.0]
    model = binvox.Voxels(data, dims, translate, 1.0, 'xyz')
    with open(filename, 'wb') as f:
        model.write(f)

def extract_name(filename):
    head, tail = os.path.split(filename)
    name, ext = os.path.splitext(tail)
    return name

def start_index(invdata):
    indices = np.transpose(np.nonzero(invdata[0, :, :]))
    if indices.shape[0] > 0:
        y, z = indices[0]
        return [0, y, z]

    indices = np.transpose(np.nonzero(invdata[:, 0, :]))
    if indices.shape[0] > 0:
        x, z = indices[0]
        return [x, 0, z]

    indices = np.transpose(np.nonzero(invdata[:, :, 0]))
    if indices.shape[0] > 0:
        x, y = indices[0]
        return [x, y, 0]

    indices = np.transpose(np.nonzero(invdata[-1, :, :]))
    if indices.shape[0] > 0:
        y, z = indices[0]
        return [-1, y, z]

    indices = np.transpose(np.nonzero(invdata[:, -1, :]))
    if indices.shape[0] > 0:
        x, z = indices[0]
        return [x, -1, z]

    indices = np.transpose(np.nonzero(invdata[:, :, -1]))
    if indices.shape[0] > 0:
        x, y = indices[0]
        return [x, y, -1]

    return None
