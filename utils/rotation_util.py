import torch
import math


# [B, n]
def normalize_vector( v, return_mag =False):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v

# u, v [B, n]
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out

def compute_rotation_matrix_from_ortho6d(ortho6d): # parameter range: -inf ~ inf
    if ortho6d.shape.__len__() != 2:
        reshape=True
    else:
        reshape=False
    if reshape:
        # [B', J, 6] -> [B, 3]
        batch_size = ortho6d.shape[0]
        num_joints = ortho6d.shape[1]
        ortho6d = ortho6d.view(-1, 6)

    x_raw = ortho6d[:,0:3]  # [B,3]
    y_raw = ortho6d[:,3:6]  # [B,3]

    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x,y,z), 2)  # [B, 3, 3]

    if reshape:
        # [B, 3, 3] -> [B', J, 3, 3]
        matrix = matrix.view(batch_size, num_joints, 3, 3)
    return matrix
    
#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 3.1416 radian (0 to 180 degree if degrees=True) batch
#snippet from github.com/papagina/RotationContinuity
def compute_geodesic_distance_from_two_matrices(m1, m2, degrees=False):
    m2 = m2.float()
    if m1.shape.__len__() != 3:
        reshape = True
    else:
        reshape = False
    if reshape:
        # [B', J, 3, 3] -> [B, 3, 3]
        batch_size = m1.shape[0]
        num_joints = m1.shape[1]
        m1 = m1.view(-1, 3, 3)
        m2 = m2.view(-1, 3, 3)

    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    theta = torch.acos(cos)
    if reshape:
        # [B] -> [B', J]
        theta = theta.view(batch_size, num_joints)
    if degrees:
        theta = theta * (180 / math.pi)
    return theta

def compute_L2_distance_from_two_matrices(m1, m2):
    # L2 loss : mean sum of squares of all matrix elements, is different from matrix L2-norm
    """
    [B, J, 3, 3], [B, J, 3, 3] -> [B, ]
    """
    B, J = m1.shape[:2]
    return ((m1 - m2)**2).view(B, -1).mean(1)  
    # ref: https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/shapenet/code/Model_pointnet.py#L97
