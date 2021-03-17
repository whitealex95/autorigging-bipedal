import os
import math
from os import listdir, mkdir
from os.path import join, exists
from tqdm import tqdm

project_path = '/home/whitealex95/Projects/autorigging'
obj_path = join(project_path, 'mixamo/objs')
character_list = sorted(listdir(obj_path))


print(character_list)
__import__('pdb').set_trace()
# character_list = [character for character in character_list if not character.startswith('Ch')]
character_list = ['Ch15_nonPBR']
# character_list = ['Ch34_nonPBR', 'Ch35_nonPBR', 'Ch36_nonPBR', 'Ch39_nonPBR', 'Ch40_nonPBR', 'Ch42_nonPBR', 'Ch44_nonPBR', 'Ch45_nonPBR', 'Ch46_nonPBR']
for character in tqdm(character_list):
	obj_list = sorted([obj for obj in listdir(os.path.join(obj_path, character)) if obj.endswith('.obj')])
	for obj in obj_list:
		source_path = join(obj_path, character, obj)
		if not os.path.exists(source_path.split('.obj')[0]+'.binvox'):
			try:
				source_path = join(obj_path, character, obj)
# command = "xvfb-run -s '-screen 0 640x480x24' ./binvox -d 82 -pb '{}'".format(source_path)
				command = "./binvox -d 82 -pb '{}'".format(source_path)
				print(command)
				os.system(command)
			except:
				__import__('pdb').set_trace()
