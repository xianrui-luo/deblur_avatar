import torch
import torch.nn as nn
import numpy as np
from .deformer import get_deformer,get_deformer_motion,get_deformer_wononrigid
from .pose_correction import get_pose_correction
from .texture import get_texture,get_texture_nonrigid
from scene.blur_kernel import smpl_net




class GaussianConverter(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata
        self.pose_diff_max=None
        self.pose_diff_min=None

        # if self.training:
        if 'cubic' not in cfg.model.pose_correction.name:
            cfg.model.pose_correction.name='interpolation'
        cfg.model.pose_correction.non_rigid_iteration=cfg.model.deformer.non_rigid.delay
 
        self.pose_correction = get_pose_correction(cfg.model.pose_correction, metadata)

        # cfg.model.deformer.name='motion'
        self.deformer = get_deformer_motion(cfg.model.deformer, metadata)

        cfg.model.texture.non_rigid_iteration=cfg.model.deformer.non_rigid.delay
        cfg.model.texture.non_rigid=cfg.model.deformer.non_rigid
        self.texture = get_texture(cfg.model.texture, metadata)
        
        # self.pose_perturb=smpl_net()

        self.optimizer, self.scheduler = None, None
        self.set_optimizer()

    
    def set_optimizer(self):
        opt_params = [
            {'params': self.deformer.rigid.parameters(), 'lr': self.cfg.opt.get('rigid_lr', 0.)},
            # {'params': self.deformer.non_rigid.parameters(), 'lr': self.cfg.opt.get('non_rigid_lr', 0.)},
            {'params': [p for n, p in self.deformer.non_rigid.named_parameters() if 'latent' not in n],
             'lr': self.cfg.opt.get('non_rigid_lr', 0.)},
            {'params': [p for n, p in self.deformer.non_rigid.named_parameters() if 'latent' in n],
             'lr': self.cfg.opt.get('nr_latent_lr', 0.), 'weight_decay': self.cfg.opt.get('latent_weight_decay', 0.05)},
            {'params': self.pose_correction.parameters(), 'lr': self.cfg.opt.get('pose_correction_lr', 0.)},
            {'params': [p for n, p in self.texture.named_parameters() if 'latent' not in n],
             'lr': self.cfg.opt.get('texture_lr', 0.)},
            {'params': [p for n, p in self.texture.named_parameters() if 'latent' in n],
             'lr': self.cfg.opt.get('tex_latent_lr', 0.), 'weight_decay': self.cfg.opt.get('latent_weight_decay', 0.05)},
        ]
        self.optimizer = torch.optim.Adam(params=opt_params, lr=0.001, eps=1e-15)

        gamma = self.cfg.opt.lr_ratio ** (1. / self.cfg.opt.iterations)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def forward(self, gaussians, camera, iteration, compute_loss=True,num_moments=5):
        loss_reg = {}

        camera_list,angle_list = self.pose_correction(camera, iteration,num_moments)
        camera_original=camera
        
        # if camera_list is a list
        if isinstance(camera_list, list):
            number=len(camera_list)
        else:
            number=1
            camera_list=[camera_list]

        # pose augmentation delete only for now
        pose_noise = self.cfg.pipeline.get('pose_noise', 0.)
        for i in range(number):
            if self.training and pose_noise > 0 and np.random.uniform() <= 0.5:
                camera = camera_list[i]
                camera=camera.copy()
                camera.rots = camera.rots + torch.randn(camera.rots.shape, device=camera.rots.device) * pose_noise
                camera_list[i]=camera
        if self.training and pose_noise > 0 and np.random.uniform() <= 0.5:
            camera = camera_original.copy()
            camera.rots = camera.rots + torch.randn(camera.rots.shape, device=camera.rots.device) * pose_noise
            camera_original=camera
            # else:
            #     camera = camera_list[i]
        
        gaussians_original, deformed_gaussians,loss_reg_deformer,xyz_list,rotation_list,pos_delta,scales_delta,rotations_delta= self.deformer(gaussians, camera_list,camera_original, iteration, compute_loss)

        loss_reg.update(loss_reg_deformer)

        
        
        color_precompute = self.texture(deformed_gaussians, [camera_original],iteration)


        
        return gaussians_original,deformed_gaussians,loss_reg, color_precompute,xyz_list,rotation_list,scales_delta,rotations_delta,pos_delta,angle_list


    def optimize(self):
        grad_clip = self.cfg.opt.get('grad_clip', 0.)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
