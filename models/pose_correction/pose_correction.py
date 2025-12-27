import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.spatial.transform import Rotation

import models
from .lbs import lbs
# from models.network_utils import get_mlp
# import pypose as pp

from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle
import roma
import spline
from scipy.interpolate import CubicSpline

def quaternion_slerp(q1, q2, alpha):
    """
    Perform Spherical Linear Interpolation (SLERP) between two quaternions.
    
    Parameters:
    - q1: Tensor of shape (batch_size, 4) representing the starting quaternion.
    - q2: Tensor of shape (batch_size, 4) representing the ending quaternion.
    - alpha: A float or tensor between 0 and 1 representing the interpolation factor.
    
    Returns:
    - interpolated_quat: Tensor of shape (batch_size, 4) representing the interpolated quaternion.
    """
    # Ensure quaternions are normalized
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)

    # Compute the dot product (cosine of the angle between the two quaternions)
    dot_product = torch.sum(q1 * q2, dim=-1, keepdim=True)

    # Ensure the shortest path is taken (quaternion symmetry: q and -q represent the same rotation)
    q2 = torch.where(dot_product < 0, -q2, q2)
    dot_product = torch.sum(q1 * q2, dim=-1, keepdim=True)

    # Clamp dot_product to avoid numerical issues with acos
    # dot_product = torch.clamp(dot_product, -1.0, 1.0)
    dot_product = torch.clamp(dot_product, -1.0+1e-6, 1.0-1e-6)

    # Calculate the angle theta between the two quaternions
    theta_0 = torch.acos(dot_product)  # theta_0 is the angle between q1 and q2
    sin_theta_0 = torch.sin(theta_0)

    # Avoid division by zero for very small angles (when sin(theta_0) is close to 0)
    small_angle = sin_theta_0 < 1e-6

    # SLERP formula
    sin_theta = torch.sin(alpha * theta_0)
    sin_theta_inv = torch.sin((1.0 - alpha) * theta_0)

    slerp_quat = (sin_theta_inv / (sin_theta_0+1e-6)) * q1 + (sin_theta / (sin_theta_0+1e-6)) * q2

    # # For very small angles, fall back to linear interpolation
    # slerp_quat = torch.where(small_angle, (1.0 - alpha) * q1 + alpha * q2, slerp_quat)

    return slerp_quat



def interpolate_pose_quaternion(pose_start, pose_end, alpha=0.5):
    """
    Interpolate between two SMPL poses using SLERP (via quaternions).
    
    Parameters:
    - pose_start: Tensor of shape (1, 72) representing the starting pose in axis-angle.
    - pose_end: Tensor of shape (1, 72) representing the ending pose in axis-angle.
    - alpha: A float between 0 and 1 representing the interpolation factor.
    
    Returns:
    - interpolated_pose: Tensor of shape (1, 72) representing the interpolated pose in axis-angle.
    """
    interpolated_pose = pose_start.clone()

    # number=pose_start.shape[0]

    # for i in range(24):  # There are 24 joints in SMPL
    # for i in range(number):  # There are 24 joints in SMPL
        # Extract the axis-angle for the current joint (3 values)
        # joint_start = pose_start[:, 3 * i: 3 * (i + 1)]
        # joint_end = pose_end[:, 3 * i: 3 * (i + 1)]
    joint_start=pose_start.view(-1,3)
    joint_end=pose_end.view(-1,3)
        
    # Convert axis-angle to quaternion
    q_start = axis_angle_to_quaternion(joint_start)
    q_end = axis_angle_to_quaternion(joint_end)
    
    # Perform SLERP interpolation
    quat_interpolated = quaternion_slerp(q_start, q_end, alpha)
    
    # Convert the interpolated quaternion back to axis-angle
    joint_interpolated = quaternion_to_axis_angle(quat_interpolated)
    interpolated_pose=joint_interpolated.view(1,-1)
        
        # Set the interpolated values back into the pose vector
        # interpolated_pose[:, 3 * i: 3 * (i + 1)] = joint_interpolated

    return interpolated_pose

from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
def interpolate_pose_cubic_bspline(poses, num_interpolated_frames=10):
    """
    Interpolate between a sequence of SMPL poses using Cubic B-Spline.
    
    Parameters:
    - poses: Tensor of shape (N, 72) representing N poses in axis-angle format.
    - num_interpolated_frames: Number of interpolated frames to generate between each pair of input poses.
    
    Returns:
    - interpolated_poses: List of tensors, each of shape (1, 72) representing the interpolated poses in axis-angle format.
    """
    num_poses = poses.shape[0]
    
    # Create an array representing the pose indices
    t = torch.linspace(0, num_poses - 1, steps=num_poses, device=poses.device)
    
    # Create an array for the target interpolated indices
    t_interp = torch.linspace(0, num_poses - 1, steps=(num_poses - 1) * num_interpolated_frames + num_poses, device=poses.device)
    
    # Calculate cubic spline coefficients
    coeffs = natural_cubic_spline_coeffs(t, poses)
    spline = NaturalCubicSpline(coeffs)
    
    # Evaluate the spline at the interpolated points
    interpolated_poses = spline.evaluate(t_interp)
    
    # Convert interpolated poses to a list of tensors of shape (1, 72)
    interpolated_poses = [interpolated_poses[i:i+1] for i in range(interpolated_poses.shape[0])]
    
    return interpolated_poses



def interpolate_pose_quaternion_roma(pose_start, pose_end, alpha=0.5):
    interpolated_pose = pose_start.clone()
    joint_start=pose_start.view(-1,3)
    joint_end=pose_end.view(-1,3)
    quat_start = roma.rotvec_to_unitquat(joint_start)
    quat_end = roma.rotvec_to_unitquat(joint_end)
    alpha=torch.tensor([alpha]).to(quat_start.device)
    #convert from double to float
    alpha=alpha.float()
    quat_end = quat_end.float()
    quat_start = quat_start.float()
    quat_interpolated = roma.utils.unitquat_slerp(quat_start, quat_end, alpha)
    joint_interpolated = roma.unitquat_to_rotvec(quat_interpolated)
    interpolated_pose=joint_interpolated.view(1,-1)
    return interpolated_pose

def get_transforms_02v(Jtr):
    device = Jtr.device

    from scipy.spatial.transform import Rotation as R
    rot45p = torch.tensor(R.from_euler('z', 45, degrees=True).as_matrix(), dtype=torch.float32, device=device)
    rot45n = torch.tensor(R.from_euler('z', -45, degrees=True).as_matrix(), dtype=torch.float32, device=device)
    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 4, 4).repeat(24, 1, 1)

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    R_02v_l = []
    t_02v_l = []
    chain = [1, 4, 7, 10]
    rot = rot45p
    for i, j_idx in enumerate(chain):
        R_02v_l.append(rot)
        t = Jtr[j_idx]
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent]
            t = torch.matmul(rot, t - t_p)
            t = t + t_02v_l[i-1]

        t_02v_l.append(t)

    R_02v_l = torch.stack(R_02v_l, dim=0)
    t_02v_l = torch.stack(t_02v_l, dim=0)
    t_02v_l = t_02v_l - torch.matmul(Jtr[chain], rot.transpose(0, 1))

    R_02v_l = F.pad(R_02v_l, (0, 0, 0, 1))  # 4 x 4 x 3
    t_02v_l = F.pad(t_02v_l, (0, 1), value=1.0)   # 4 x 4

    bone_transforms_02v[chain] = torch.cat([R_02v_l, t_02v_l.unsqueeze(-1)], dim=-1)

    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    R_02v_r = []
    t_02v_r = []
    chain = [2, 5, 8, 11]
    rot = rot45n
    for i, j_idx in enumerate(chain):
        # bone_transforms_02v[j_idx, :3, :3] = rot
        R_02v_r.append(rot)
        t = Jtr[j_idx]
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent]
            t = torch.matmul(rot, t - t_p)
            t = t + t_02v_r[i-1]

        t_02v_r.append(t)

    # bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    R_02v_r = torch.stack(R_02v_r, dim=0)
    t_02v_r = torch.stack(t_02v_r, dim=0)
    t_02v_r = t_02v_r - torch.matmul(Jtr[chain], rot.transpose(0, 1))

    R_02v_r = F.pad(R_02v_r, (0, 0, 0, 1))  # 4 x 3
    t_02v_r = F.pad(t_02v_r, (0, 1), value=1.0)   # 4 x 4

    bone_transforms_02v[chain] = torch.cat([R_02v_r, t_02v_r.unsqueeze(-1)], dim=-1)

    return bone_transforms_02v


# def calculate_pose_difference(rots1, rots2, Jtrs1, Jtrs2, bone_transforms1, bone_transforms2, w_rot=1.0, w_jtrs=1.0, w_bone=1.0):
def calculate_pose_difference(rots1, rots2, Jtrs1, Jtrs2, w_rot=1.0, w_jtrs=1.0, w_bone=1.0):
    # # 1. Rotation Difference
    # def rotation_difference(r1, r2):
    #     # Reshape rotations to (batch_size, num_joints, 3, 3)
    #     r1_mat = r1.reshape(r1.shape[0], r1.shape[1], 3, 3)
    #     r2_mat = r2.reshape(r2.shape[0], r2.shape[1], 3, 3)
        
    #     # # Calculate the difference using matrix multiplication
    #     # diff = torch.matmul(r1_mat.transpose(-1, -2), r2_mat)
    #     # # Convert to a measure of difference (Frobenius norm)
    #     # diff_norm = torch.norm(diff - torch.eye(3, device=diff.device).unsqueeze(0).unsqueeze(0), dim=(-2, -1))
    #     diff = r1_mat - r2_mat
        
    #     # Compute the Frobenius norm
    #     diff_norm = torch.norm(diff.reshape(diff.shape[0], diff.shape[1], -1), dim=-1)
        # return diff_norm

    # rot_diff = rotation_difference(rots1, rots2)
    # # rot_diff_norm = rot_diff.mean()
    # rot_diff_norm = rot_diff.mean()
    rot_diff=rots2-rots1
    rot_diff_norm=(rot_diff**2).mean()

    # 2. Joint Transformation Difference
    jtrs_diff = Jtrs2 - Jtrs1
    # jtrs_diff_norm = torch.norm(jtrs_diff, dim=-1).mean()
    jtrs_diff_norm = (jtrs_diff**2).mean()

    # 3. Bone Transform Difference
    # bone_diff = bone_transforms2 - bone_transforms1
    # # bone_diff_norm = torch.norm(bone_diff.reshape(bone_diff.shape[0], -1), dim=-1).mean()
    # bone_diff_norm = (bone_diff**2).mean()

    # Combine the differences
    # total_diff = w_rot * rot_diff_norm + w_jtrs * jtrs_diff_norm + w_bone * bone_diff_norm
    total_diff = w_rot * rot_diff_norm + w_jtrs * jtrs_diff_norm

    # return total_diff, rot_diff_norm, jtrs_diff_norm, bone_diff_norm
    return total_diff



def bone_transform_diff_to_tensor(transforms1, transforms2):
    """
    Calculate the difference between two sets of bone transforms and return it as a PyTorch tensor.
    
    :param transforms1: PyTorch tensor of shape [24, 4, 4] representing the first set of bone transforms
    :param transforms2: PyTorch tensor of shape [24, 4, 4] representing the second set of bone transforms
    :return: PyTorch tensor representing the differences (shape: [24, 12])
    """
    # Ensure inputs are PyTorch tensors
    t1 = transforms1.float() if isinstance(transforms1, torch.Tensor) else torch.from_numpy(transforms1).float()
    t2 = transforms2.float() if isinstance(transforms2, torch.Tensor) else torch.from_numpy(transforms2).float()
    
    # Extract rotation matrices (3x3 upper-left submatrix) for all 24 poses
    epsilon = 1e-4
    R1 = t1[:, :3, :3]
    R2 = t2[:, :3, :3]
    
    # Calculate rotation differences
    R_diff = torch.matmul(R2, R1.transpose(1, 2))
    # trace = torch.trace(R_diff)  # Trace of the rotation difference matrix
    # cos_theta = (trace - 1) / 2
    # cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # To avoid numerical issues
    # rotation_diff = torch.acos(cos_theta)
    
    # # Convert rotation differences to axis-angle representation
    # angle = torch.acos((R_diff.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).unsqueeze(-1)
    cos_angle = (R_diff.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1 + epsilon, 1 - epsilon)  # Avoid domain errors
    angle = torch.acos(cos_angle)

    axis = torch.stack([
        R_diff[:, 2, 1] - R_diff[:, 1, 2],
        R_diff[:, 0, 2] - R_diff[:, 2, 0],
        R_diff[:, 1, 0] - R_diff[:, 0, 1]
    ], dim=-1)
    # axis = axis / (2 * torch.sin(angle+1e-6))
    axis_norm = torch.sqrt(torch.sum(axis**2, dim=-1, keepdim=True) + epsilon)
    axis_normalized = axis / axis_norm
    angle_factor = torch.sin(angle / 2) / (angle / 2 + epsilon)
    rotation_diff = 2 * axis_normalized * angle.unsqueeze(-1) * angle_factor.unsqueeze(-1)

    rotation_angles=torch.norm(rotation_diff,dim=-1)
    max_angle=torch.max(rotation_angles,dim=0)[0]
    
    # Combine axis and angle into a single vector
    # rotation_diff = axis * angle
    
    # Extract translation vectors
    # translation1 = t1[:, :3, 3]
    # translation2 = t2[:, :3, 3]
    
    # Calculate translation differences
    # translation_diff = translation2 - translation1
    
    # Combine rotation and translation differences into a single feature tensor
    # feature_tensor = torch.cat([rotation_diff, translation_diff], dim=-1)
    
    return max_angle




class NoPoseCorrection(nn.Module):
    def __init__(self, config, metadata=None):
        super(NoPoseCorrection, self).__init__()

    def forward(self, camera, iteration):
        return camera, {}

    def regularization(self, out):
        return {}

class PoseCorrection(nn.Module):
    def __init__(self, config, metadata=None):
        super(PoseCorrection, self).__init__()

        self.config = config
        self.metadata = metadata

        self.frame_dict = metadata['frame_dict']

        gender = metadata['gender']

        v_template = np.load('body_models/misc/v_templates.npz')[gender]
        lbs_weights = np.load('body_models/misc/skinning_weights_all.npz')[gender]
        posedirs = np.load('body_models/misc/posedirs_all.npz')[gender]
        posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
        shapedirs = np.load('body_models/misc/shapedirs_all.npz')[gender]
        J_regressor = np.load('body_models/misc/J_regressors.npz')[gender]
        kintree_table = np.load('body_models/misc/kintree_table.npy')

        self.register_buffer('v_template', torch.tensor(v_template, dtype=torch.float32).unsqueeze(0))
        self.register_buffer('posedirs', torch.tensor(posedirs, dtype=torch.float32))
        self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=torch.float32))
        self.register_buffer('J_regressor', torch.tensor(J_regressor, dtype=torch.float32))
        self.register_buffer('lbs_weights', torch.tensor(lbs_weights, dtype=torch.float32))
        self.register_buffer('kintree_table', torch.tensor(kintree_table, dtype=torch.int32))

    def forward_smpl(self, betas, root_orient, pose_body, pose_hand, trans):
        full_pose = torch.cat([root_orient, pose_body, pose_hand], dim=-1)
        verts_posed, Jtrs_posed, Jtrs, bone_transforms, _, v_posed, v_shaped, rot_mats = lbs(betas=betas,
                                                                                             pose=full_pose,
                                                                                             v_template=self.v_template.clone(),
                                                                                             clothed_v_template=None,
                                                                                             shapedirs=self.shapedirs.clone(),
                                                                                             posedirs=self.posedirs.clone(),
                                                                                             J_regressor=self.J_regressor.clone(),
                                                                                             parents=self.kintree_table[
                                                                                                 0].long(),
                                                                                             lbs_weights=self.lbs_weights.clone(),
                                                                                             dtype=torch.float32)

        rots = torch.cat([torch.eye(3).reshape(1, 1, 3, 3).to(rot_mats.device), rot_mats[:, 1:]], dim=1)
        rots = rots.reshape(1, -1, 9).contiguous()

        bone_transforms_02v = get_transforms_02v(Jtrs.squeeze(0))

        bone_transforms = torch.matmul(bone_transforms.squeeze(0), torch.inverse(bone_transforms_02v))
        bone_transforms[:, :3, 3] = bone_transforms[:, :3, 3] + trans

        v_shaped = v_shaped.detach()
        center = torch.mean(v_shaped, dim=1)
        minimal_shape_centered = v_shaped - center
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05

        # compute pose condition
        Jtrs = Jtrs - center
        Jtrs = (Jtrs - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtrs -= 0.5
        Jtrs *= 2.
        Jtrs = Jtrs.contiguous()

        verts_posed = verts_posed + trans[None]

        return rots, Jtrs, bone_transforms, verts_posed, v_posed, Jtrs_posed

    # def forward(self, camera, iteration):
    def forward(self, camera, iteration,num_moments):
        frame = camera.frame_id
        if frame not in self.frame_dict:
            return camera, {}
        return self.pose_correct(camera, iteration,num_moments)

    def regularization(self, out):
        raise NotImplementedError

    def pose_correct(self, camera, iteration):
        raise NotImplementedError

class DirectPoseOptimization(PoseCorrection):
    def __init__(self, config, metadata=None):
        super(DirectPoseOptimization, self).__init__(config, metadata)
        self.cfg = config

        root_orient = metadata['root_orient']
        pose_body = metadata['pose_body']
        pose_hand = metadata['pose_hand']
        trans = metadata['trans']
        betas = metadata['betas']
        frames = metadata['frames']

        self.frames = frames

        # use nn.Embedding
        root_orient = np.array(root_orient)
        pose_body = np.array(pose_body)
        pose_hand = np.array(pose_hand)
        trans = np.array(trans)
        self.root_orients = nn.Embedding.from_pretrained(torch.from_numpy(root_orient).float(), freeze=False)
        self.pose_bodys = nn.Embedding.from_pretrained(torch.from_numpy(pose_body).float(), freeze=False)
        self.pose_hands = nn.Embedding.from_pretrained(torch.from_numpy(pose_hand).float(), freeze=False)
        self.trans = nn.Embedding.from_pretrained(torch.from_numpy(trans).float(), freeze=False)

        self.register_parameter('betas', nn.Parameter(torch.tensor(betas, dtype=torch.float32)))

    def pose_correct(self, camera, iteration):
        if iteration < self.cfg.get('delay', 0):
            return camera, {}

        frame = camera.frame_id

        # use nn.Embedding
        idx = torch.Tensor([self.frame_dict[frame]]).long().to(self.betas.device)
        root_orient = self.root_orients(idx)
        pose_body = self.pose_bodys(idx)
        pose_hand = self.pose_hands(idx)
        trans = self.trans(idx)

        betas = self.betas

        # compose rots, Jtrs, bone_transforms, posed_smpl_verts
        rots, Jtrs, bone_transforms, posed_smpl_verts, _, _ = self.forward_smpl(betas, root_orient, pose_body, pose_hand, trans)

        rots_diff = camera.rots - rots
        updated_camera = camera.copy()
        updated_camera.update(
            rots=rots,
            Jtrs=Jtrs,
            bone_transforms=bone_transforms,
        )

        loss_pose = (rots_diff ** 2).mean()
        return updated_camera, {
            'pose': loss_pose,
        }

    def regularization(self, out):
        loss = (out['rots_diff'] ** 2).mean()
        return {'pose_reg': loss}

    def export(self, frame):
        model_dict = {}

        idx = torch.Tensor([self.frame_dict[frame]]).long().to(self.betas.device)
        root_orient = self.root_orients(idx)
        pose_body = self.pose_bodys(idx)
        pose_hand = self.pose_hands(idx)
        trans = self.trans(idx)

        betas = self.betas

        rots, Jtrs, bone_transforms, posed_smpl_verts, v_posed, Jtr_posed = self.forward_smpl(betas, root_orient, pose_body,
                                                                                pose_hand, trans)
        model_dict.update({
            'minimal_shape': v_posed[0],
            'betas': betas,
            'Jtr_posed': Jtr_posed[0],
            'bone_transforms': bone_transforms,
            'trans': trans[0],
            'root_orient': root_orient[0],
            'pose_body': pose_body[0],
            'pose_hand': pose_hand[0],
        })
        for k, v in model_dict.items():
            model_dict.update({k: v.detach().cpu().numpy()})
        return model_dict

class PoseInterpolation(PoseCorrection):
    # def __init__(self, config, metadata=None,num_momoents=4):
    def __init__(self, config, metadata=None):
        super(PoseInterpolation, self).__init__(config, metadata)
        self.cfg = config

        root_orient = metadata['root_orient']
        pose_body = metadata['pose_body']
        pose_hand = metadata['pose_hand']
        trans = metadata['trans']
        betas = metadata['betas']
        frames = metadata['frames']

        self.pose_body_original=pose_body
        self.pose_hand_original=pose_hand

        self.frames = frames

        # use nn.Embedding
        root_orient = np.array(root_orient)
        pose_body = np.array(pose_body)
        pose_hand = np.array(pose_hand)
        trans = np.array(trans)

        

        # add noise to the pose
        pose_body_start = pose_body + np.random.normal(0,0.05,pose_body.shape)
        # pose_body_start = pose_body
        pose_hand_start = pose_hand + np.random.normal(0,0.05,pose_hand.shape)
        # root_orient_start = root_orient + np.random.normal(0,0.05,root_orient.shape)
        root_orient_start = root_orient
        # trans_start=trans+np.random.normal(0,0.05,root_orient.shape)



        self.root_orients = nn.Embedding.from_pretrained(torch.from_numpy(root_orient).float(), freeze=False)
        self.root_orients_original= nn.Embedding.from_pretrained(torch.from_numpy(root_orient_start).float(), freeze=False)

        self.pose_bodys = nn.Embedding.from_pretrained(torch.from_numpy(pose_body).float(), freeze=False)
        self.pose_bodys_original = nn.Embedding.from_pretrained(torch.from_numpy(pose_body_start).float(), freeze=False)
        self.pose_hands = nn.Embedding.from_pretrained(torch.from_numpy(pose_hand).float(), freeze=False)
        self.pose_hands_original = nn.Embedding.from_pretrained(torch.from_numpy(pose_hand_start).float(), freeze=False)
        self.trans = nn.Embedding.from_pretrained(torch.from_numpy(trans).float(), freeze=False)
        self.trans_original = nn.Embedding.from_pretrained(torch.from_numpy(trans).float(), freeze=False)





        self.register_parameter('betas', nn.Parameter(torch.tensor(betas, dtype=torch.float32)))
        self.max_angle_list={}

    def pose_correct(self, camera, iteration, num_moments=5):

        ##delete for debugging
        if iteration < self.cfg.get('delay', 0):
            # return [camera], {}
            return [camera],self.max_angle_list
 

        frame = camera.frame_id

        # use nn.Embedding
        idx = torch.Tensor([self.frame_dict[frame]]).long().to(self.betas.device)
        root_orient = self.root_orients(idx)
        root_orient_original = self.root_orients_original(idx)
        pose_body = self.pose_bodys(idx)
        pose_body_original = self.pose_bodys_original(idx)
        pose_hand = self.pose_hands(idx)
        pose_hand_original = self.pose_hands_original(idx)
        trans = self.trans(idx)
        trans_original=self.trans_original(idx)

        


        betas = self.betas

        

        ##interpolate pose with the alpha from 0 to 1
        interpolated_poses_body = []
        interpolated_poses_hand = []
        interpolated_root_orient = []
        interpolated_trans = []
        # if self.training:
        for alpha in np.linspace(0, 1, num_moments):
            interpolated_pose_body = interpolate_pose_quaternion(pose_body_original, pose_body, alpha)
            interpolated_pose_hand = interpolate_pose_quaternion(pose_hand_original, pose_hand, alpha)
            interpolated_root_orient.append(interpolate_pose_quaternion(root_orient_original, root_orient, alpha))
            interpolated_poses_body.append(interpolated_pose_body)
            interpolated_poses_hand.append(interpolated_pose_hand)
            interpolated_trans.append(trans_original*(1-alpha)+alpha*trans)


        rots_list=[]
        Jtrs_list=[]
        bone_transforms_list=[]
        # posed_smpl_verts_list=[]
        updated_camera_list = []

        # rots, Jtrs, bone_transforms, _, _, _ = self.forward_smpl(betas, root_orient, pose_body, pose_hand,trans)
        # bone_transforms_new=self.bone_transforms(idx)
        

        for i in range(num_moments):
            # compose rots, Jtrs, bone_transforms, posed_smpl_verts
            rots, Jtrs, bone_transforms, _, _, _ = self.forward_smpl(betas, interpolated_root_orient[i], interpolated_poses_body[i], interpolated_poses_hand[i],interpolated_trans[i])
            rots_list.append(rots)
            Jtrs_list.append(Jtrs)
            bone_transforms_list.append(bone_transforms)
            updated_camera= camera.copy()
            updated_camera.update(
                rots=rots,
                Jtrs=Jtrs,
                bone_transforms=bone_transforms,
            )
            updated_camera_list.append(updated_camera)
            rots_diff = camera.rots - rots


            if i==(num_moments-1)//2:
                Jtrs_diff=camera.Jtrs-Jtrs
                loss_pose = (rots_diff ** 2).mean()+ (Jtrs_diff ** 2).mean()
        
        # pose_diff=bone_transform_diff_to_tensor(bone_transforms_list[0],bone_transforms_list[-1])
        if iteration>=self.cfg.get('delay', 0)+2000 and self.max_angle_list=={}:
            ##get all the index and its corresponding bone_transforms_list
            frame_list=self.frame_dict
            for frame in frame_list:
                idx = torch.Tensor([frame_list[frame]]).long().to(self.betas.device)
                root_orient = self.root_orients(idx)
                pose_body = self.pose_bodys(idx)
                pose_hand = self.pose_hands(idx)
                root_orient_original = self.root_orients_original(idx)
                pose_body_original = self.pose_bodys_original(idx)
                pose_hand_original = self.pose_hands_original(idx)
                trans = self.trans(idx)
                trans_original=self.trans_original(idx)

                _, _, bone_transforms_max, _, _, _ = self.forward_smpl(betas, root_orient, pose_body,pose_hand,trans)
                _, _, bone_transforms_min, _, _, _ = self.forward_smpl(betas, root_orient_original, pose_body_original,pose_hand_original,trans_original)

                max_angle=bone_transform_diff_to_tensor(bone_transforms_min,bone_transforms_max)
                self.max_angle_list[int(idx)]=max_angle






        return updated_camera_list,self.max_angle_list
        # return updated_camera_list, {
        #     'pose': loss_pose,
        # }

    def regularization(self, out):
        loss = (out['rots_diff'] ** 2).mean()
        return {'pose_reg': loss}

    