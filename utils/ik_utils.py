import torch
from utils.joint_util import transform_rel2glob

'''
    rotation should have shape batch_size * Joint_num * (3/4) * Time
    position should have shape batch_size * 3 * Time
    offset should have shape batch_size * Joint_num * 3
    output have shape batch_size * Time * Joint_num * 3
'''

'''
Input:
    joint_position: (B, J, 3) (Time x)
    offset: (B, J, 3)
Output: 
    rotation: (B, J, (3/4/6)) (Time x)
'''
from utils.bvh_utils import get_animated_bvh_joint_positions


class InverseKinematics:
    def __init__(self, rotations, positions, offset, parents):
        #         self.quater = args.rotation == 'quaternion'
        if rotations is None:
            self.rotations = torch.rand(positions.shape[:-1] + (4,), device=positions.device)  # random quaternion
        else:
            self.rotations = rotations  # local bvh rotation
        self.rotations.requires_grad_(True)

        self.positions = positions  # global joint position
        self.root_position = self.positions[..., 0, :]
        self.offset = offset
        self.parents = parents  # topology info

        # Optimizers for IK
        self.optimizer = torch.optim.Adam([self.rotations], lr=1e-2, betas=(0.9, 0.999))
        self.crit = torch.nn.MSELoss()

    def step(self):
        self.optimizer.zero_grad()
        positions_fk = self.forward(self.rotations, self.root_position, self.offset, order='', quater=True, world=True)
        self.loss = loss = self.crit(positions_fk, self.positions)
        loss.backward()
        self.optimizer.step()
        self.positions_fk = positions_fk
        return loss.item()

    '''
        rotation should have shape batch_size * Joint_num * (3/4) * Time
        position should have shape batch_size * 3 * Time
        offset should have shape batch_size * Joint_num * 3
        output have shape batch_size * Time * Joint_num * 3
    '''

    def forward_(self):
        return self.forward(self.rotations, self.root_position, self.offset, order='', quater=True, world=True)

    def forward(self, rotation: torch.Tensor, root_position: torch.Tensor, offset: torch.Tensor, order='xyz',
                quater=False,
                world=True):

        if quater:  # Default rotation representation is quaternion. You should try something else though
            transform = self.transform_from_quaternion(rotation)  # B, J, 3, 3
            transform_glob = transform_rel2glob(transform)
        else:
            raise NotImplementedError
        return get_animated_bvh_joint_positions(offset, transform_glob, root_position)

    @staticmethod
    def transform_from_axis(euler, axis):
        transform = torch.empty(euler.shape[0:3] + (3, 3), device=euler.device)
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform

    @staticmethod
    def transform_from_quaternion(quater: torch.Tensor):

        norm = torch.norm(quater, dim=-1, keepdim=True)
        quater = quater / norm

        #  Fucking problem here... Fixed from (w, x, y, z) -> (x, y, z, w)
        qx = quater[..., 0]
        qy = quater[..., 1]
        qz = quater[..., 2]
        qw = quater[..., 3]

        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        m = torch.empty(quater.shape[:-1] + (3, 3), device=quater.device)
        m[..., 0, 0] = 1.0 - (yy + zz)
        m[..., 0, 1] = xy - wz
        m[..., 0, 2] = xz + wy
        m[..., 1, 0] = xy + wz
        m[..., 1, 1] = 1.0 - (xx + zz)
        m[..., 1, 2] = yz - wx
        m[..., 2, 0] = xz - wy
        m[..., 2, 1] = yz + wx
        m[..., 2, 2] = 1.0 - (xx + yy)

        return m

