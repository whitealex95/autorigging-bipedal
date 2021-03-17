import torch
from utils.joint_util import transform_rel2glob
from torch_geometric.data import Batch

def mesh_transform(mesh, JM, BM, skin_weight, J=22, JM_type='rel', BM_type='glob'):
    # Todo: transform vertex using V_transformed = skin_weight * BM * JM^-1 * V
    """
    Input:
        vertex: torch.Tensor [B, npts, 3] --> (jsbae) it must be raw input, i.e. without sampling applied
        JM: torch.Tensor [B, J, 4, 4] --> (capoo) Don't forget it's global! (kim)No, it's relative....
        BM: torch.Tensor [B, J, 4, 4] --> (capoo) Don't forget it's global!
        skin_weight: torch.Tensor [B, npts, J]
    Output:
        transformed_vertex
    """
    vertex_batch, batch = mesh.pos, mesh.batch
    B = JM.shape[0]  # batch_size

    if JM_type != 'glob':
        JM_global = transform_rel2glob(JM)
    else:
        JM_global = JM
    IJM = []
    for i in range(B):
        IJM.append(torch.stack([JM_j.inverse() for JM_j in JM_global[i]], dim=0).view(J, 4, 4))
    IJM = torch.stack(IJM, dim=0)
    transformed_vertex_batch = torch.zeros_like(vertex_batch)
    for batch_idx in range(B):
        vertex = vertex_batch[batch==batch_idx]
        V = vertex.__len__()
        vertex = torch.cat((vertex, torch.ones(V, 1).to(vertex.device)), dim=-1)  # vertx: [B, npts, 4]
        local_vertex = torch.einsum('kij,vj->kvi', IJM[batch_idx], vertex)
        BM_transformed = torch.einsum('kij,kvj->kvi', BM[batch_idx], local_vertex)
        transformed_vertex = torch.einsum('vk,kvi->vi', skin_weight[batch==batch_idx], BM_transformed)
        transformed_vertex_batch[batch==batch_idx] = transformed_vertex[:, :3]
    transformed_mesh = Batch(pos=transformed_vertex_batch, edge_index=mesh.edge_index, batch=batch)

    return transformed_mesh

def vertex_transform(vertex, JM, BM, skin_weight, J=22, JM_type='rel', BM_type='glob'):
    # Todo: transform vertex using V_transformed = skin_weight * BM * JM^-1 * V
    """
    Input:
        vertex: torch.Tensor [B, npts, 3] --> (jsbae) it must be raw input, i.e. without sampling applied
        JM: torch.Tensor [B, J, 4, 4] --> (capoo) Don't forget it's global! (kim)No, it's relative....
        BM: torch.Tensor [B, J, 4, 4] --> (capoo) Don't forget it's global!
        skin_weight: torch.Tensor [B, npts, J]
    Output:
        transformed_vertex
    """
    if len(vertex.shape) == 2:
        B = 1
        V = vertex.shape[0]
        vertex = vertex.view(B, V, 3)
        JM = JM.view(B, J, 4, 4)
        BM = BM.view(B, J, 4, 4)
        skin_weight = skin_weight.view(B, V, J)
    else:
        B, V, _ = vertex.shape

    vertex = torch.cat((vertex, torch.ones(B, V, 1).to(vertex.device)), dim=-1)  # vertx: [B, npts, 4]
    if JM_type != 'glob':
        JM_global = transform_rel2glob(JM)
    else:
        JM_global = JM
    IJM = []
    for i in range(B):
        IJM.append(torch.stack([JM_j.inverse() for JM_j in JM_global[i]], dim=0).view(J, 4, 4))
    IJM = torch.stack(IJM, dim=0)
    local_vertex = torch.einsum('bkij,bvj->bkvi', IJM, vertex)
    BM_transformed = torch.einsum('bkij,bkvj->bkvi', BM, local_vertex)
    transformed_vertex = torch.einsum('bvk,bkvi->bvi', skin_weight, BM_transformed)

    return transformed_vertex[...,:3]
