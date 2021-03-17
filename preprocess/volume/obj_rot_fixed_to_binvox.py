import os
import math
from os import listdir, mkdir
from os.path import join, exists
from tqdm import tqdm
import subprocess

data_dir = '/home/whitealex95/Projects/autorigging/autorigging/data/mixamo'
# characters_list = sorted(os.listdir(os.path.join(data_dir, 'animated')))
characters_list = ['Ch24_nonPBR', 'Ch29_nonPBR', 'kaya',
                  'maria_j_j_ong', 'paladin_j_nordstrom', 'pumpkinhulk_l_shaw']
characters_list = ['Ch14_nonPBR']
print(characters_list)
# motions_list = sorted([f.split('.obj')[0] for f in os.listdir(os.path.join(data_dir, 'objs', characters_list[0])) if
#                        f.endswith('.obj') and f != 'bindpose.obj'])
motions_list = ['Back Squat', 'Drunk Walk Backwards', 'Samba Dancing', 'Shoved Reaction With Spin_1',
           'Shoved Reaction With Spin_2', 'Warming Up_1', 'Warming Up_2']
# #TestKim
# motions_list = ['Back Squat']
# motions_list = ['\ '.join(motion.split(' ')) for motion in motions_list]
print(motions_list)

frame_counts = [18, 13, 274, 45, 
                45, 95, 95]


for character in characters_list:
	for motion_idx, motion in enumerate(motions_list):
		for frame_number in range(frame_counts[motion_idx]):
			frame_name = '%s_%06d'%(motion, frame_number)

			obj_filename = data_dir + '/objs_fixed/'+character+'/%s.obj'%(frame_name)

			try:
				# command = "xvfb-run -s '-screen 0 640x480x24' ./binvox -d 82 -pb '{}'".format(source_path)
				command = "./binvox -d 82 -pb '{}'".format(obj_filename)
				print(command)
				os.system(command)
				# subprocess.Popen(command)
			except:
				__import__('pdb').set_trace()
