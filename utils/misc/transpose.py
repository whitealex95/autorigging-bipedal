import numpy as np

num_joints = 52 #TODO
filename = 'inference_tree_test'#TODO
T=np.genfromtxt('./test/'+filename+'.csv', delimiter=',')
newT = [None] * num_joints

for i, A in enumerate(T):
    A = np.reshape(A, (3,4))
    R = A[:, :3]
    newR = np.transpose(R)
    newA = np.reshape(np.append(newR, np.expand_dims(A[:, 3], axis=-1), axis=1), (12,))
    newT[i] = newA

newT = np.array(newT)
np.savetxt('./new_'+filename+'.csv', newT, delimiter=',')



