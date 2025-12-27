import torch.nn as nn

from models.deformer.rigid import get_rigid_deform
from models.deformer.non_rigid import get_non_rigid_deform

class Deformer(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.rigid = get_rigid_deform(cfg.rigid, metadata)
        self.non_rigid = get_non_rigid_deform(cfg.non_rigid, metadata)

    def forward(self, gaussians, camera, iteration, compute_loss=True,delay_flag=False):
        loss_reg = {}
        deformed_gaussians, loss_non_rigid = self.non_rigid(gaussians, iteration, camera, compute_loss,delay_flag)
        deformed_gaussians_rigid,xyz_list,rotation_list = self.rigid(deformed_gaussians, iteration, camera)
        if iteration>self.cfg.non_rigid.delay:
            deformed_gaussians_onlyrigid,xyz_list_onlyrigid,rotation_list_onlyrigid = self.rigid(gaussians, iteration, camera)
        else:
            deformed_gaussians_onlyrigid=deformed_gaussians_rigid
            xyz_list_onlyrigid=xyz_list
            rotation_list_onlyrigid=rotation_list


        loss_reg.update(loss_non_rigid)
        return deformed_gaussians, loss_reg,xyz_list,rotation_list,xyz_list_onlyrigid,rotation_list_onlyrigid,deformed_gaussians_rigid,deformed_gaussians_onlyrigid
class Deformer_motion(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        cfg.rigid.name='motion'
        self.rigid = get_rigid_deform(cfg.rigid, metadata)
        cfg.non_rigid.name='motion'
        self.non_rigid = get_non_rigid_deform(cfg.non_rigid, metadata)

    def forward(self, gaussians, camera, camera_original,iteration, compute_loss=True):
        loss_reg = {}
        gaussians,_,pos_delta,scale_delta,rotation_delta = self.non_rigid(gaussians, iteration, camera, compute_loss)
        deformed_gaussians,xyz_list,rotation_list = self.rigid(gaussians, iteration, camera, camera_original,pos_delta,scale_delta,rotation_delta)
        return gaussians,deformed_gaussians, loss_reg,xyz_list,rotation_list,pos_delta,scale_delta,rotation_delta
    

    

def get_deformer(cfg, metadata):
    return Deformer(cfg, metadata)

def get_deformer_motion(cfg, metadata):
    return Deformer_motion(cfg, metadata)
