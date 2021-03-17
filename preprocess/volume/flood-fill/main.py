import numpy as np
import glob
import os
import util
import time
from os import listdir, mkdir
from os.path import join, exists
from tqdm import tqdm

# paths = glob.glob("path/to/*.binvox")
# paths = glob.glob("../data/aj.binvox")
# print("number of data", len(paths))

project_path = '/home/whitealex95/Projects/autorigging'
obj_path = join(project_path, 'mixamo_4k/objs')
character_list = sorted(listdir(obj_path))

for character in tqdm(character_list):
    obj_list = sorted([obj for obj in listdir(os.path.join(obj_path, character)) if obj.endswith('.obj')])
    for obj in obj_list:
        path = join(obj_path, character, obj.split('.obj')[0] + '.binvox')

        data = util.read_binvox(path)
        invdata = np.ones(data.shape, dtype=np.bool) & ~data
        start = util.start_index(invdata)

        xdim, ydim, zdim = data.shape
        fill_value = False
        while start is not None:
            old_value = invdata[start[0], start[1], start[2]]
            stack = set([(start[0], start[1], start[2])])

            if fill_value == old_value:
                raise ValueError("Filling region with same value")

            while stack:
                x, y, z = stack.pop()
                if invdata[x, y, z] == old_value:
                    invdata[x, y, z] = fill_value
                    if x > 0:
                        stack.add((x - 1, y, z))
                    if x < (xdim - 1):
                        stack.add((x + 1, y, z))
                    if y > 0:
                        stack.add((x, y - 1, z))
                    if y < (ydim - 1):
                        stack.add((x, y + 1, z))
                    if z > 0:
                        stack.add((x, y, z - 1))
                    if z < (zdim - 1):
                        stack.add((x, y, z + 1))

            start = util.start_index(invdata)

        data += invdata
        util.save_binvox(path, data)  # overwrite previous binvox
        # util.save_binvox(path.split('.binvox')[0] + '_ff.binvox' , data)