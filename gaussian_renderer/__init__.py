#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import pytorch3d.transforms as tf


from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import math
def draw_init_points(scene, background):
    with torch.no_grad():
        # Set field of view
        FoVx, FoVy = 1, 1
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)

        scaling_modifier = 1.0  # Increased scaling for better visualization

        # Define world to view transformation for front view
        world_view_transform_front = torch.tensor([
            [1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [0., 0., 2., 1.]
        ]).cuda()

        # Define the projection matrix
        projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0, 1).cuda()

        # Compute the full transformation matrix
        full_proj_transform_front = (world_view_transform_front.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        
        # Get camera center
        camera_center_front = world_view_transform_front.inverse()[3, :3]

        # Set rasterization settings
        raster_settings_front = GaussianRasterizationSettings(
            image_height=512,  # Increased resolution for better visualization
            image_width=512,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=scaling_modifier,
            viewmatrix=world_view_transform_front,
            projmatrix=full_proj_transform_front,
            sh_degree=0,
            campos=camera_center_front,
            prefiltered=False,
            debug=False
        )

        # Create rasterizer
        rasterizer = GaussianRasterizer(raster_settings=raster_settings_front)

        # Get initial Gaussian positions
        init_xyz = scene.gaussians.get_xyz
        means3D = init_xyz

        # Optionally, you could add visualization code here if needed
        return rasterizer, means3D

def render_pose(data,
           iteration,
           scene,
           pipe,  
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           compute_loss=True,
           return_opacity=False, ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

# /
    num_momoents=9
    
    pc,pc_deform_original,loss_reg, colors_precomp,xyz_list,rotation_list,scales_delta,rotations_delta,pos_delta,angle_list = scene.convert_gaussians_test(data, iteration, compute_loss,num_momoents)
   
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(data.FoVx * 0.5)
    tanfovy = math.tan(data.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data.image_height),
        image_width=int(data.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data.world_view_transform,
        projmatrix=data.full_proj_transform,
        # sh_degree=pc.active_sh_degree,
        sh_degree=pc.active_sh_degree,
        campos=data.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    renders = []
    radiis = []
    visibility_filters = []
    viewspace_points = []
    opacity_images = []


    opacity_original=None

    # for i in range(len(pc)):
    #     # print(i)    
    # means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
  
    lambda_p=0.01
    lambda_s=0.01
    max_clamp=1.1
    
    scale_i=pc_deform_original.get_scaling
    rotation_final=pc_deform_original.rotation_precomp
    cov3D_precomp_original = pc_deform_original.get_covariance(scaling_modifier,scale_i,rotation_final)

    shs = None
    scales = None
    rotations = None
    
    rendered_original, radii = rasterizer(
    means3D = pc_deform_original.get_xyz,
    means2D = means2D,
    shs = shs,
    colors_precomp = colors_precomp,
    opacities = opacity,
    scales = scales,
    rotations = rotations,
    cov3D_precomp = cov3D_precomp_original)

    if return_opacity:
        opacity_original, _ = rasterizer(
            means3D=pc_deform_original.get_xyz,
            means2D=means2D,
            shs=None,
            colors_precomp=torch.ones(opacity.shape[0], 3, device=opacity.device),
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp_original)
        opacity_original = opacity_original[:1]

    for i in range(len(xyz_list)):
        pc_=pc.clone()
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2D = screenspace_points

        pc_._xyz=xyz_list[i]
        scale_i=pc_.get_scaling
        rotation_final=rotation_list[i]
        
       

        if pipe.compute_cov3D_python:
            
            cov3D_precomp = pc_.get_covariance(scaling_modifier,scale_i,rotation_final)
            
        shs = None


        means3D_=pc_.get_xyz

        rendered_image, radii = rasterizer(
        means3D = means3D_,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    
        renders.append(rendered_image)
        # viewspace_points_pc = [screenspace_points]
        viewspace_points.append(screenspace_points)
        visibility_filters.append(radii > 0)
        radiis.append(radii)


        if return_opacity:
            opacity_image, _ = rasterizer(
                means3D=means3D_,
                means2D=means2D,
                shs=None,
                colors_precomp=torch.ones(opacity.shape[0], 3, device=opacity.device),
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp)
            opacity_image = opacity_image[:1]
            opacity_images.append(opacity_image)

    renders.append(rendered_original)
    opacity_images.append(opacity_original)
    rendered_image = sum(renders) / len(renders)
    opacity_final=None
    if return_opacity:
        opacity_final=opacity_images[0]
        if len(opacity_images)>1:
            for i in range(1,len(opacity_images)):
                opacity_final=1-(1-opacity_final)*(1-opacity_images[i])

    return {"deformed_gaussian": pc_deform_original,
            "render": rendered_image,
            "render_original": rendered_original,
            "viewspace_points": viewspace_points,
            "visibility_filter" : visibility_filters,
            "radii": radiis,
            "loss_reg": loss_reg,
            "opacity_render": opacity_final,
            "angle_list":angle_list,

            }
