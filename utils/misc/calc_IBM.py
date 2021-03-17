import os
import numpy as np

def main():
    for f in os.listdir("../test/gt/transforms"):
        print(f)
        convert(f)


def convert(filename):
    num_joints = 22 #TODO
    tree =maketree(num_joints)
    T0 = np.genfromtxt('./test/gt/transforms/'+filename, delimiter=',')
    T = [None] * num_joints
    BM = [None]*num_joints
    IBM = [None] * num_joints

    for i,A in enumerate(T0):
        A = np.reshape(np.append(A,[0., 0., 0., 1.]),(4,4))
        T[i] = A

    bfs(tree, T, BM)
    BM = np.array(BM)

    for i,A in enumerate(BM):
        IBM[i] = np.linalg.inv(A)

    IBM = np.array(IBM)
    IBM = np.reshape(IBM, (num_joints, 16))
    np.savetxt('./test/gt/transforms/IBM_'+filename, IBM, delimiter=',')
    return

def bfs(child, T, BM):
    def bfs_rec(child, T, BM, node):
        for x in child[node]:
            BM[x] = np.matmul(BM[node],T[x])
        for x in child[node]:
            bfs_rec(child, T, BM, x)
        return
    BM[0] = T[0]
    bfs_rec(child, T, BM, 0)
    return

def maketree(num_joints):
    # hardcoded tree
    if num_joints==52:
        child = [[] for x in range(num_joints)]
        child[0] = [1, 44, 48]
        child[1] = [2]
        child[2] = [3]
        child[3] = [4, 6, 25]
        child[4] = [5]
        #child[5]
        child[6] = [7]
        child[7] = [8]
        child[8] = [9]
        child[9] = [10, 13, 16, 19, 22]
        child[10] = [11]
        child[11] = [12]
        #child[12]
        child[13] = [14]
        child[14] = [15]
        #child[15]
        child[16] = [17]
        child[17] = [18]
        #child[18]
        child[19] = [20]
        child[20] = [21]
        #child[21]
        child[22] = [23]
        child[23] = [24]
        #child[24]
        child[25] = [26]
        child[26] = [27]
        child[27] = [28]
        child[28] = [29, 32, 35, 38, 41]
        child[29] = [30]
        child[30] = [31]
        #child[31]
        child[32] = [33]
        child[33] = [34]
        #child[34]
        child[35] = [36]
        child[36] = [37]
        #child[37]
        child[38] = [39]
        child[39] = [40]
        #child[40]
        child[41] = [42]
        child[42] = [43]
        #child[43]
        child[44] = [45]
        child[45] = [46]
        child[46] = [47]
        #child[47]
        child[48] = [49]
        child[49] = [50]
        child[50] = [51]
        #child[51]
        return child
    else:
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

if __name__=="__main__":
    main()
