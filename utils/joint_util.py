import torch
import numpy as np

# Numpy Function
def bfs(child, T, T_glob):
    if torch.is_tensor(T):
        def bfs_rec(child, T, T_glob, node):
            for x in child[node]:
                T_glob[x] = torch.matmul(T_glob[node],T[x])
            for x in child[node]:
                bfs_rec(child, T, T_glob, x)
            return

    else:
        def bfs_rec(child, T, T_glob, node):
            for x in child[node]:
                T_glob[x] = np.matmul(T_glob[node],T[x])
            for x in child[node]:
                bfs_rec(child, T, T_glob, x)
            return
    T_glob[0] = T[0]
    bfs_rec(child, T, T_glob, 0)
    return

# 22 joints
joint_names = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm',
               'RightShoulder', 'RightArm', 'RightForeArm', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
               'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
               'LeftHand', 'RightHand']
# 21 bones
bone_names = ['Hips-Spine', 'Hips-LeftUpLeg', 'Hips-RightUpLeg', 'Spine-Spine1', 'Spine1-Spine2', 'Spine2-Neck', 'Spine2-LeftShoulder', 'Spine2-RightShoulder', 'Neck-Head', 'LeftShoulder-LeftArm', 'LeftArm-LeftForeArm', 'LeftForeArm-LeftHand', 'RightShoulder-RightArm', 'RightArm-RightForeArm', 'RightForeArm-RightHand', 'LeftUpLeg-LeftLeg', 'LeftLeg-LeftFoot', 'LeftFoot-LeftToeBase', 'RightUpLeg-RightLeg', 'RightLeg-RightFoot', 'RightFoot-RightToeBase']

def joint_pos2bone_len(joint_pos):
    if len(joint_pos.shape) == 2:
        J, _ = joint_pos.shape
        B = 0
        joint_pos = joint_pos.unsqueeze(0)
    else:
        B, J, _ = joint_pos.shape
    tree = maketree(J)
    bone_lengths = []
    for node_idx, child_list in enumerate(tree):
        for child_idx in child_list:
            bone_lengths.append(torch.sqrt(torch.sum((joint_pos[:, node_idx] - joint_pos[:, child_idx])**2, dim=-1, keepdim=True)))  # BxJ
    bone_lengths = torch.cat(bone_lengths, dim=1)
    if B==0:
        bone_lengths = bone_lengths.squeeze(0)
    return bone_lengths

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
    else: # joint 22
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

# numpy function
def modify_joint_matrix(transforms, num_joints, inverse=True, local2global=True, short=False):
    tree =maketree(num_joints)
    T0 = transforms
    T = [None] * num_joints
    JM = [None] * num_joints
    IJM = [None] * num_joints

    for i,A in enumerate(T0):
        if A.shape[-2:] == (4, 4):
            pass
        elif A.shape[-1] == 12:
            A = np.reshape(np.append(A,[0., 0., 0., 1.]),(4,4))
        T[i] = A

    if local2global:
        bfs(tree, T, JM)
        JM = np.array(JM)
    else:
        JM = np.array(T)
    if inverse:
        for i,A in enumerate(JM):
            IJM[i] = np.linalg.inv(A)

        IJM = np.array(IJM)
        IJM = np.reshape(IJM, (num_joints, 16))
        if short:
            return IJM[:,:12]
        return IJM

    JM = np.reshape(JM, (num_joints, 16))
    if short:
        return JM[:, :12]
    return JM

# torch functions
def transform_rel2glob(transforms):
    batch_size = transforms.shape[0]
    num_joints = transforms.shape[1]

    tree = maketree(num_joints)

    # glob_transforms = torch.zeros(transforms.shape).to(transforms.device).type(transforms.type())
    glob_transforms = []
    for batch_idx, T in enumerate(transforms):
        # T_glob = glob_transforms[batch_idx]
        T_glob = [None] * num_joints
        bfs(tree, T, T_glob)
        glob_transforms.append(torch.stack(T_glob).to(transforms.device).type(transforms.type()))
    return torch.stack(glob_transforms).to(transforms.device).type(transforms.type())


def toSE3(R_or_T3x4: torch.Tensor, p: torch.Tensor=None)->torch.Tensor:
    """
    R: torch.Tensor (batch_size, num_joints, 3, 3) or (num_joints, 3, 3)
    p: torch.Tensor (batch_size, num_joints, 3) or (num_joints, 3)
    """
    is_batched = R_or_T3x4.shape.__len__() == 4
    if is_batched:
        batch_size = R_or_T3x4.shape[0]
        num_joints = R_or_T3x4.shape[1]
    else:
        batch_size = 1
        num_joints = R_or_T3x4.shape[0]

    if R_or_T3x4.shape[-1] == 3:
        R = R_or_T3x4
        if not is_batched:
            R = R.unsqueeze(0)
            p = p.unsqueeze(0)
        T3x4 = torch.cat([R, p.unsqueeze(-1)], axis=-1)

    elif R_or_T3x4.shape[-1] == 4:
        T3x4 = R_or_T3x4
        if not is_batched:
            T3x4 = T3x4.unsqueeze(0)
    else:
        raise NotImplementedError

    T =torch.cat([T3x4, torch.Tensor([0, 0, 0, 1] * batch_size * num_joints).type(T3x4.type()).view(batch_size, num_joints, 1, 4)], axis=-2)

    if not is_batched:
        T = T.squeeze(0)

    return T



# numpy function
def get_transform_matrix(rot6d, trans, num_joints=22):
    a1 = rot6d[:, :3]
    a2 = rot6d[:, 3:]
    b1 = a1 / np.sqrt(np.reshape(np.sum(a1**2, axis=1), (num_joints, 1)))
    b2 = a2 - np.reshape(np.sum(np.multiply(a2, b1), axis=1), (num_joints, 1)) * b1
    b2 = b2 / np.sqrt(np.reshape(np.sum(b2**2, axis=1), (num_joints,1)))
    b3 = np.cross(b1, b2)

    b1 = np.expand_dims(b1, axis=2)
    b2 = np.expand_dims(b2, axis=2)
    b3 = np.expand_dims(b3, axis=2)

    R = np.concatenate((b1,b2,b3), axis=2)# (num_joints, 3, 3)
    T = np.expand_dims(trans, axis=2) # (num_joints, 3, 1)

    transform = np.reshape(np.concatenate((R,T), axis=2), (num_joints, 12)) # (args.num_joints, 12)

    return transform

def transform_joint(joint_path="./data/mixamo/transforms/aj/Samba Dancing_000101.csv"):
    transforms = np.genfromtxt(joint_path, delimiter=',', dtype=float)
    IJM = modify_joint_matrix(transforms, num_joints=22, inverse=False, local2global=True, short=True)
    print("Saving IJM to ",joint_path.split('.csv')[0]+ '_IJM.csv' )
    np.savetxt(joint_path.split('.csv')[0]+ '_JM_GT.csv', IJM, delimiter=',')
    return 

# numpy function
def save_ijm(rot6d, trans, output_filename, joint_loss_type):
    # rot = compute_rotation_matrix_from_ortho6d(rot6d)
    transform = get_transform_matrix(rot6d, trans)
    local2global = joint_loss_type in ['rel', 'rel2glob']  # Todo: verify
    inverse_joint_matrix = modify_joint_matrix(transform, num_joints=22, inverse=True,
                                               local2global=local2global, short=False)
    np.savetxt(output_filename, inverse_joint_matrix, delimiter=',')

def save_jm(rot6d, trans, output_filename, joint_loss_type):
    # rot = compute_rotation_matrix_from_ortho6d(rot6d)
    transform = get_transform_matrix(rot6d, trans)
    local2global = joint_loss_type in ['rel', 'rel2glob']  # Todo: verify
    joint_matrix = modify_joint_matrix(transform, num_joints=22, inverse=False,
                                               local2global=local2global, short=False)
    np.savetxt(output_filename, joint_matrix, delimiter=',')

def save_jm2(jm, output_filename, joint_type='glob', inverse=False):
    jm = modify_joint_matrix(jm, num_joints=22, inverse=inverse,
                                               local2global=joint_type != 'global', short=False)
    np.savetxt(output_filename, jm, delimiter=', ')


if __name__ == "__main__":
    # __import__('pdb').set_trace()
    transform_joint("./data/mixamo/transforms/aj/Samba Dancing_000000.csv")