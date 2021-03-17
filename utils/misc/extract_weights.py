import os
import sys
import numpy as np
BASE_DIR=os.path.abspath(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '/pycollada'))
from collada import *

inputRepoPath = '~/autorigging/obj_dae'
outputRepoPath = '~/autorigging/weights/sht/'
joint_num = 22

indices = [*range(1,10)]
#wrongIndices = [1,3,7,8,11,14]


#for x in wrongIndices:
#    indices.remove(x)

for modIdx in indices:
    fileName = '%s/%d/Boxing.dae'%(inputRepoPath, modIdx)
    mesh = Collada(fileName)
    output = np.zeros((4096,joint_num))

    joint_dictionary={}
    for joint in range(joint_num-2):
        jointname = mesh.animations[joint].id[:-5]
        joint_dictionary[jointname]=joint
    joint_dictionary['mixamorig_LeftHand']=20
    joint_dictionary['mixamorig_RightHand']=21

    for idx in range(0,4096):
        count = 0
        for jidx in mesh.controllers[0].joint_index[idx]:
            jointname = mesh.controllers[0].weight_joints[jidx]
            correct_jidx = joint_dictionary[jointname]
            output[idx][correct_jidx] = mesh.controllers[0].weights[mesh.controllers[0].weight_index[idx][count]]
            count=count+1
    print(outputRepoPath+'sht'+str(modIdx)+'.csv')
    np.savetxt(outputRepoPath+'sht'+ str(modIdx).zfill(2)+'.csv', output, delimiter=',')
