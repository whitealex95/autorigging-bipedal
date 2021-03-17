import os
import sys
import numpy as np
BASE_DIR=os.path.abspath(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '/pycollada'))
from os import listdir
from os.path import join
from collada import *

project_path = '/home/whitealex95/Projects/autorigging'
animation_path = join(project_path, 'mixamo/animated_4k')
transforms_path = join(project_path, 'mixamo/transforms_4k')


indices = [*range(1,10)]
#wrongIndices = [1, 3, 7, 8, 14]
#for x in wrongIndices:
#    indices.remove(x)

joint_num = 22
action = ["Back Squat", "Drunk Walk Backwards", "Samba Dancing", "Shoved Reaction With Spin_1", "Shoved Reaction With Spin_2", "Warming Up_1", "Warming Up_2"]
# fCount = [15, 11, 220, 37, 37, 77, 77]
fCount = [18, 13, 274, 45, 45, 95, 95]
end_string = 'Matrix-animation-output-transform'

animation_models_list = sorted(listdir(animation_path))
for animation_model in animation_models_list:
    animations_list = sorted([ anim for anim in listdir(join(animation_path, animation_model)) if anim.endswith('.dae')])

    if not os.path.exists(join(transforms_path, animation_model)):
        os.makedirs(join(transforms_path, animation_model))
    
    for animation in animations_list:
        source_path = join(animation_path, animation_model, animation)
        dest_path = lambda frame: join(transforms_path, animation_model, animation.split('.dae')[0] + '_' + str(frame).zfill(6)+'.csv')
        dest_ibm_path = join(transforms_path, animation_model, 'ibm.csv')

        mesh = Collada(source_path)
        if mesh.controllers[0].max_joint_index != joint_num - 1:
            # print("number of joint(shape index: "+ str(modIdx) + f") is not {joint_num}.")
            __import__('pdb').set_trace()
            continue

        for frame in range(fCount[action.index(animation.split('.dae')[0])]):
            output = np.zeros((joint_num, 12))
            ibm = np.zeros((joint_num, 12))
            for joint in range(joint_num):
                if 0<= joint and joint<=19: 
                    animIdx = mesh.animations[joint].id[:-4] + end_string
                    transform_matrices = mesh.animations[joint].sourceById[animIdx].data
                    transform_matrices = transform_matrices.reshape(-1, 16)
                    output[joint] = transform_matrices[frame][:12]
                    ibm[joint] = mesh.controllers[0].joint_matrices[animIdx.split('-')[0]].reshape(16,)[:12]
                elif joint==20: # mixamorig_LeftHand
                    try:
                        parentIBM = mesh.controllers[0].joint_matrices['mixamorig_LeftForeArm']
                        myIBM = mesh.controllers[0].joint_matrices['mixamorig_LeftHand']
                    except:
                        parentIBM = mesh.controllers[0].joint_matrices['boss_LeftForeArm']
                        myIBM = mesh.controllers[0].joint_matrices['boss_LeftHand']
                    output[joint] = np.matmul(parentIBM, np.linalg.inv(myIBM)).reshape(16,)[:12]
                    ibm[joint] = myIBM.reshape(16,)[:12]
                elif joint==21: # mixamorig_RightHand
                    try:
                        parentIBM = mesh.controllers[0].joint_matrices['mixamorig_RightForeArm']
                        myIBM = mesh.controllers[0].joint_matrices['mixamorig_RightHand']
                    except:
                        parentIBM = mesh.controllers[0].joint_matrices['boss_RightForeArm']
                        myIBM = mesh.controllers[0].joint_matrices['boss_RightHand']
                    output[joint] = np.matmul(parentIBM, np.linalg.inv(myIBM)).reshape(16,)[:12]
                    ibm[joint] = myIBM.reshape(16,)[:12]
            print("[Saving JM] ", dest_path(frame), end='\r')
            np.savetxt(dest_path(frame), output, delimiter=',')
    print('[Saving IBM] ', dest_ibm_path)
    np.savetxt(dest_ibm_path, ibm, delimiter=',')

def maketree(num_joints=22):
    # we assume that ,
    # INDEX 20 -> mixamorig_LeftHand
    # INDEX 21 -> mixamorig_RightHan
    child = [[] for x in range(num_joints)]
    child[0] = [1, 12, 16]
    child[1] = [2]
    child[2] = [3]
    child[3] = [4, 6, 9]
    child[4] = [5]
    #child[5]
    child[6] = [7]
    child[7] = [8]
    child[8] = [20] ##### ATTENTION
    child[9] = [10]
    child[10] = [11]
    child[11] = [21] ##### ATTENTION
    child[12]=[13]
    child[13]=[14]
    child[14]=[15]
    #child[15]
    child[16]=[17]
    child[17]=[18]
    child[18]=[19]
    #child[19]
    
    return child
