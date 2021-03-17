import os
import sys
import numpy as np
BASE_DIR=os.path.abspath('')
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '/pycollada'))
from os import listdir, mkdir
from os.path import join, exists
from collada import *

project_path = '/home/whitealex95/Projects/autorigging'
animation_path = join(project_path, 'mixamo/animated_4k')
vertex_weight_path = join(project_path, 'mixamo/weights_4k')
joint_cnt = 22

animation_models_list = sorted(listdir(animation_path))
animation_models_list = ['zombie']
for animation_model in animation_models_list:
    animations_list = sorted([ anim for anim in listdir(join(animation_path, animation_model)) if anim.endswith('.dae')])
    animation = animations_list[0]

    source_path = join(animation_path, animation_model, animation)
    dest_path = join(vertex_weight_path, animation_model+'.csv')

    mesh = Collada(source_path)
    vertex_cnt = mesh.controllers[0].vcounts.__len__()
    output = np.zeros((vertex_cnt, joint_cnt))

    joint_dictionary = {}
    for joint in range(joint_cnt - 2):
        jointname = mesh.animations[joint].id.split('_')[1][:-5]
        joint_dictionary[jointname]=joint  # assign joint number for each joint name
    joint_dictionary['LeftHand']=20
    joint_dictionary['RightHand']=21
    # joint_dictionary['mixamorig_LeftHand']=20
    # joint_dictionary['mixamorig_RightHand']=21

    for idx in range(vertex_cnt):
        count = 0
        for jidx in mesh.controllers[0].joint_index[idx]:
            jointname = mesh.controllers[0].weight_joints[jidx].split('_')[1]
            correct_jidx = joint_dictionary[jointname]
            output[idx][correct_jidx] = mesh.controllers[0].weights[mesh.controllers[0].weight_index[idx][count]]
            count = count + 1
    print(dest_path)
    np.savetxt(dest_path, output, delimiter=',')
