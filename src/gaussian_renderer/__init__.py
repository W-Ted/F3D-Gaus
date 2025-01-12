# Adapted from https://github.com/graphdeco-inria/gaussian-splatting/tree/main
# to take in a predicted dictionary with 3D Gaussian parameters.

import math
import torch
import numpy as np

# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from diff_surfel_rasterization import GaussianRasterizationSettings_2DGS, GaussianRasterizer_2DGS
from diff_gof_rasterization import GaussianRasterizationSettings_GOF, GaussianRasterizer_GOF
# from diff_gof_rasterization_fov import GaussianRasterizationSettings_GOF_FOV, GaussianRasterizer_GOF_FOV
# from utils.graphics_utils import focal2fov


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def focal2fov_torch(focal, pixels):
    return 2*torch.atan(pixels/(2*focal))

# def render_predicted(pc : dict, 
#                      world_view_transform,
#                      full_proj_transform,
#                      camera_center,
#                      bg_color : torch.Tensor, 
#                      cfg, 
#                      scaling_modifier = 1.0, 
#                      override_color = None,
#                      focals_pixels = None):
#     """
#     Render the scene as specified by pc dictionary. 
    
#     Background tensor (bg_color) must be on GPU!
#     """
 
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     screenspace_points = torch.zeros_like(pc["xyz"], dtype=pc["xyz"].dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     if focals_pixels == None:
#         tanfovx = math.tan(cfg.data.fov * np.pi / 360)
#         tanfovy = math.tan(cfg.data.fov * np.pi / 360)
#     else:
#         tanfovx = math.tan(0.5 * focal2fov(focals_pixels[0].item(), cfg.data.training_resolution))
#         tanfovy = math.tan(0.5 * focal2fov(focals_pixels[1].item(), cfg.data.training_resolution))

#     # Set up rasterization configuration
#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(cfg.data.training_resolution),
#         image_width=int(cfg.data.training_resolution),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=world_view_transform,
#         projmatrix=full_proj_transform,
#         sh_degree=cfg.model.max_sh_degree,
#         campos=camera_center,
#         prefiltered=False,
#         debug=False
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = pc["xyz"]
#     means2D = screenspace_points
#     opacity = pc["opacity"]

#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None

#     scales = pc["scaling"]
#     rotations = pc["rotation"]

#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#     shs = None
#     colors_precomp = None
#     if override_color is None:
#         if "features_rest" in pc.keys():
#             shs = torch.cat([pc["features_dc"], pc["features_rest"]], dim=1).contiguous()
#         else:
#             shs = pc["features_dc"]
#     else:
#         colors_precomp = override_color

#     # Rasterize visible Gaussians to image, obtain their radii (on screen). 
#     # rendered_image, radii = rasterizer(
#     rendered_image, radii, _, _ = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = shs,
#         colors_precomp = colors_precomp,
#         opacities = opacity,
#         scales = scales,
#         rotations = rotations,
#         cov3D_precomp = cov3D_precomp)

#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     return {"render": rendered_image,
#             "viewspace_points": screenspace_points,
#             "visibility_filter" : radii > 0,
#             "radii": radii}




# def render_predicted_more(pc : dict, 
#                      world_view_transform,
#                      full_proj_transform,
#                      camera_center,
#                      bg_color : torch.Tensor, 
#                      cfg, 
#                      scaling_modifier = 1.0, 
#                      override_color = None,
#                      focals_pixels = None):
#     """
#     Render the scene as specified by pc dictionary. 
    
#     Background tensor (bg_color) must be on GPU!
#     """
 
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     screenspace_points = torch.zeros_like(pc["xyz"], dtype=pc["xyz"].dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     if focals_pixels == None:
#         tanfovx = math.tan(cfg.data.fov * np.pi / 360)
#         tanfovy = math.tan(cfg.data.fov * np.pi / 360)
#         # print('tanfovx in render init', tanfovx, tanfovy)
#     else:
#         tanfovx = math.tan(0.5 * focal2fov(focals_pixels[0].item(), cfg.data.training_resolution))
#         tanfovy = math.tan(0.5 * focal2fov(focals_pixels[1].item(), cfg.data.training_resolution))

#     # Set up rasterization configuration
#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(cfg.data.training_resolution),
#         image_width=int(cfg.data.training_resolution),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=world_view_transform,
#         projmatrix=full_proj_transform,
#         sh_degree=cfg.model.max_sh_degree,
#         campos=camera_center,
#         prefiltered=False,
#         debug=False
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = pc["xyz"]
#     means2D = screenspace_points
#     opacity = pc["opacity"]

#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None

#     scales = pc["scaling"]
#     rotations = pc["rotation"]

#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#     shs = None
#     colors_precomp = None
#     if override_color is None:
#         if "features_rest" in pc.keys():
#             shs = torch.cat([pc["features_dc"], pc["features_rest"]], dim=1).contiguous()
#         else:
#             shs = pc["features_dc"]
#     else:
#         colors_precomp = override_color

#     # Rasterize visible Gaussians to image, obtain their radii (on screen). 
#     rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = shs,
#         colors_precomp = colors_precomp,
#         opacities = opacity,
#         scales = scales,
#         rotations = rotations,
#         cov3D_precomp = cov3D_precomp)

#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     return {"render": rendered_image,
#             "rendered_depth": rendered_depth,
#             "rendered_alpha": rendered_alpha,
#             "viewspace_points": screenspace_points,
#             "visibility_filter" : radii > 0,
#             "radii": radii}




def render_predicted_more_v1(pc : dict, 
                     world_view_transform,
                     full_proj_transform,
                     camera_center,
                     bg_color : torch.Tensor, 
                     cfg, 
                     scaling_modifier = 1.0, 
                     override_color = None,
                     focals_pixels = None):
    """
    Render the scene as specified by pc dictionary. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc["xyz"], dtype=pc["xyz"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if focals_pixels == None:
        tanfovx = math.tan(cfg['model']['fov'] * np.pi / 360)
        tanfovy = math.tan(cfg['model']['fov'] * np.pi / 360)
        # print('tanfovx in render init', tanfovx, tanfovy)
    else:
        tanfovx = math.tan(0.5 * focal2fov(focals_pixels[0].item(), cfg['model']['training_resolution']))
        tanfovy = math.tan(0.5 * focal2fov(focals_pixels[1].item(), cfg['model']['training_resolution']))

    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=int(cfg['model']['training_resolution']),
        image_width=int(cfg['model']['training_resolution']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=cfg['model']['max_sh_degree'],
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc["xyz"]
    means2D = screenspace_points
    opacity = pc["opacity"]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    scales = pc["scaling"]
    rotations = pc["rotation"]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if "features_rest" in pc.keys():
            shs = torch.cat([pc["features_dc"], pc["features_rest"]], dim=1).contiguous()
        else:
            shs = pc["features_dc"]
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "rendered_depth": rendered_depth,
            "rendered_alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}




def render_predicted_more_v2(pc : dict, bs,
                     world_view_transform,
                     full_proj_transform,
                     camera_center,
                     bg_color : torch.Tensor, 
                     cfg, 
                     scaling_modifier = 1.0, 
                     override_color = None,
                     focals_pixels = None):
    """
    Render the scene as specified by pc dictionary. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc["xyz"][bs], dtype=pc["xyz"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if focals_pixels == None:
        tanfovx = math.tan(cfg['model']['fov'] * np.pi / 360)
        tanfovy = math.tan(cfg['model']['fov'] * np.pi / 360)
        # print('tanfovx in render init', tanfovx, tanfovy)
    else:
        tanfovx = math.tan(0.5 * focal2fov(focals_pixels[0].item(), cfg['model']['training_resolution']))
        tanfovy = math.tan(0.5 * focal2fov(focals_pixels[1].item(), cfg['model']['training_resolution']))

    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=int(cfg['model']['training_resolution']),
        image_width=int(cfg['model']['training_resolution']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=cfg['model']['max_sh_degree'],
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc["xyz"][bs]
    means2D = screenspace_points
    opacity = pc["opacity"][bs]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    scales = pc["scaling"][bs]
    rotations = pc["rotation"][bs]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if "features_rest" in pc.keys():
            shs = torch.cat([pc["features_dc"][bs], pc["features_rest"][bs]], dim=1).contiguous()
        else:
            shs = pc["features_dc"][bs]
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "rendered_depth": rendered_depth,
            "rendered_alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}



def render_predicted_more_v3(pc : dict, bs,
                     world_view_transform,
                     full_proj_transform,
                     camera_center,
                     bg_color : torch.Tensor, 
                     cfg, 
                     scaling_modifier = 1.0, 
                     override_color = None,
                     focals_pixels = None):
    """
    Render the scene as specified by pc dictionary. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc["xyz"][bs], dtype=pc["xyz"].dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros_like(pc[bs]["xyz"], dtype=pc[bs]["xyz"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if focals_pixels == None:
        tanfovx = math.tan(cfg['model']['fov'] * np.pi / 360)
        tanfovy = math.tan(cfg['model']['fov'] * np.pi / 360)
        # print('tanfovx in render init', tanfovx, tanfovy)
    else:
        tanfovx = math.tan(0.5 * focal2fov(focals_pixels[0].item(), cfg['model']['training_resolution']))
        tanfovy = math.tan(0.5 * focal2fov(focals_pixels[1].item(), cfg['model']['training_resolution']))

    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=int(cfg['model']['training_resolution']),
        image_width=int(cfg['model']['training_resolution']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=cfg['model']['max_sh_degree'],
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc["xyz"][bs]
    means3D = pc[bs]["xyz"]
    means2D = screenspace_points
    # opacity = pc["opacity"][bs]
    opacity = pc[bs]["opacity"]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    # scales = pc["scaling"][bs]
    # rotations = pc["rotation"][bs]
    scales = pc[bs]["scaling"]
    rotations = pc[bs]["rotation"]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if "features_rest" in pc[bs].keys():
            # shs = torch.cat([pc["features_dc"][bs], pc["features_rest"][bs]], dim=1).contiguous()
            shs = torch.cat([pc[bs]["features_dc"], pc[bs]["features_rest"]], dim=1).contiguous()
        else:
            # shs = pc["features_dc"][bs]
            shs = pc[bs]["features_dc"]
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "rendered_depth": rendered_depth,
            "rendered_alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}




def render_predicted_more_v2_2dgs(pc : dict, bs,
                     world_view_transform,
                     full_proj_transform,
                     camera_center,
                     bg_color : torch.Tensor, 
                     cfg, 
                     scaling_modifier = 1.0, 
                     override_color = None,
                     focals_pixels = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc["xyz"][bs], dtype=pc["xyz"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    if focals_pixels == None:
        tanfovx = math.tan(cfg['model']['fov'] * np.pi / 360)
        tanfovy = math.tan(cfg['model']['fov'] * np.pi / 360)
        # print('tanfovx in render init', tanfovx, tanfovy)
    else:
        tanfovx = math.tan(0.5 * focal2fov(focals_pixels[0].item(), cfg['model']['training_resolution']))
        tanfovy = math.tan(0.5 * focal2fov(focals_pixels[1].item(), cfg['model']['training_resolution']))

    # # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings_2DGS(
        # image_height=int(viewpoint_camera.image_height),
        # image_width=int(viewpoint_camera.image_width),
        image_height=int(cfg['model']['training_resolution']),
        image_width=int(cfg['model']['training_resolution']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        # sh_degree=pc.active_sh_degree,
        # campos=viewpoint_camera.camera_center,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=cfg['model']['max_sh_degree'],
        campos=camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer_2DGS(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # means2D = screenspace_points
    # opacity = pc.get_opacity
    means3D = pc["xyz"][bs]
    means2D = screenspace_points
    opacity = pc["opacity"][bs]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     # currently don't support normal consistency loss if use precomputed covariance
    #     splat2world = pc.get_covariance(scaling_modifier)
    #     W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
    #     near, far = viewpoint_camera.znear, viewpoint_camera.zfar
    #     ndc2pix = torch.tensor([
    #         [W / 2, 0, 0, (W-1) / 2],
    #         [0, H / 2, 0, (H-1) / 2],
    #         [0, 0, far-near, near],
    #         [0, 0, 0, 1]]).float().cuda().T
    #     world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
    #     cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    # else:
    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation
    scales = pc["scaling"][bs]
    rotations = pc["rotation"][bs]
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        # if pipe.convert_SHs_python:
        #     shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        #     dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        #     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        #     sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # else:
        #     shs = pc.get_features
        if "features_rest" in pc.keys():
            # shs = torch.cat([pc["features_dc"][bs], pc["features_rest"][bs]], dim=1).contiguous()
            shs = torch.cat([pc["features_dc"][bs], pc["features_rest"][bs]], dim=1).contiguous()
        else:
            # shs = pc["features_dc"][bs]
            shs = pc["features_dc"][bs]
    else:
        colors_precomp = override_color
    
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }


    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    # print('render_normal: ', render_normal.shape, render_normal.dtype, render_normal.min(), render_normal.max())
    # print('world_view_transform: ', world_view_transform.shape, world_view_transform.dtype, world_view_transform.min(), world_view_transform.max())
    # render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    # render_normal = (render_normal.permute(1,2,0) @ (world_view_transform[:3,:3].T)).permute(2,0,1)
    render_normal = (render_normal.permute(1,2,0) @ (world_view_transform[0,0,:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    # render_depth_median = torch.nan_to_num(render_depth_median, 0, 0, 0)

    # # get expected depth map
    # render_depth_expected = allmap[0:1]
    # render_depth_expected = (render_depth_expected / render_alpha)
    # render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    # # render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0, 0)
    
    # get depth distortion map
    # render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    # surf_depth = render_depth_expected * (1-cfg['model']['depth_ratio']) + (cfg['model']['depth_ratio']) * render_depth_median
    surf_depth = (cfg['model']['depth_ratio']) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    # surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    # surf_normal = depth_to_normal(world_view_transform, int(cfg['model']['training_resolution']), \
    surf_normal = depth_to_normal(world_view_transform[0,0], int(cfg['model']['training_resolution']), \
                                  int(cfg['model']['training_resolution']), \
                                    cfg['model']['fov'] * np.pi / 360 / 0.5, \
                                        cfg['model']['fov'] * np.pi / 360 /0.5, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()


    rets.update({
            'rendered_alpha': render_alpha,
            'rendered_normal': render_normal,
            # 'rend_dist': render_dist,
            'rendered_depth': surf_depth,
            'surf_normal': surf_normal,
    })

    return rets




def render_predicted_more_v3_2dgs(pc : dict, bs,
                     world_view_transform,
                     full_proj_transform,
                     camera_center,
                     bg_color : torch.Tensor, 
                     cfg, 
                     scaling_modifier = 1.0, 
                     override_color = None,
                     focals_pixels = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc[bs]["xyz"], dtype=pc[bs]["xyz"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    if focals_pixels == None:
        tanfovx = math.tan(cfg['model']['fov'] * np.pi / 360)
        tanfovy = math.tan(cfg['model']['fov'] * np.pi / 360)
        # print('tanfovx in render init', tanfovx, tanfovy)
    else:
        tanfovx = math.tan(0.5 * focal2fov(focals_pixels[0].item(), cfg['model']['training_resolution']))
        tanfovy = math.tan(0.5 * focal2fov(focals_pixels[1].item(), cfg['model']['training_resolution']))

    # # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings_2DGS(
        # image_height=int(viewpoint_camera.image_height),
        # image_width=int(viewpoint_camera.image_width),
        image_height=int(cfg['model']['training_resolution']),
        image_width=int(cfg['model']['training_resolution']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        # sh_degree=pc.active_sh_degree,
        # campos=viewpoint_camera.camera_center,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=cfg['model']['max_sh_degree'],
        campos=camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer_2DGS(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # means2D = screenspace_points
    # opacity = pc.get_opacity
    means3D = pc[bs]["xyz"]
    means2D = screenspace_points
    opacity = pc[bs]["opacity"]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     # currently don't support normal consistency loss if use precomputed covariance
    #     splat2world = pc.get_covariance(scaling_modifier)
    #     W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
    #     near, far = viewpoint_camera.znear, viewpoint_camera.zfar
    #     ndc2pix = torch.tensor([
    #         [W / 2, 0, 0, (W-1) / 2],
    #         [0, H / 2, 0, (H-1) / 2],
    #         [0, 0, far-near, near],
    #         [0, 0, 0, 1]]).float().cuda().T
    #     world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
    #     cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    # else:
    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation
    scales = pc[bs]["scaling"]
    rotations = pc[bs]["rotation"]
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        # if pipe.convert_SHs_python:
        #     shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        #     dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        #     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        #     sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # else:
        #     shs = pc.get_features
        if "features_rest" in pc[bs].keys():
            # shs = torch.cat([pc["features_dc"][bs], pc["features_rest"][bs]], dim=1).contiguous()
            shs = torch.cat([pc[bs]["features_dc"], pc[bs]["features_rest"]], dim=1).contiguous()
        else:
            # shs = pc["features_dc"][bs]
            shs = pc[bs]["features_dc"]
    else:
        colors_precomp = override_color
    
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }


    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    # render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    # render_normal = (render_normal.permute(1,2,0) @ (world_view_transform[:3,:3].T)).permute(2,0,1)
    render_normal = (render_normal.permute(1,2,0) @ (world_view_transform[0,0,:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    # render_depth_median = torch.nan_to_num(render_depth_median, 0, 0, 0)

    # # get expected depth map
    # render_depth_expected = allmap[0:1]
    # render_depth_expected = (render_depth_expected / render_alpha)
    # render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    # # render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0, 0)
    
    # # get depth distortion map
    # render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    # surf_depth = render_depth_expected * (1-cfg['model']['depth_ratio']) + (cfg['model']['depth_ratio']) * render_depth_median
    surf_depth = (cfg['model']['depth_ratio']) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    # surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    # surf_normal = depth_to_normal(world_view_transform, int(cfg['model']['training_resolution']), \
    surf_normal = depth_to_normal(world_view_transform[0,0], int(cfg['model']['training_resolution']), \
                                  int(cfg['model']['training_resolution']), \
                                    cfg['model']['fov'] * np.pi / 360 / 0.5, \
                                        cfg['model']['fov'] * np.pi / 360 /0.5, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()


    rets.update({
            'rendered_alpha': render_alpha,
            'rendered_normal': render_normal,
            # 'rend_dist': render_dist,
            'rendered_depth': surf_depth,
            'surf_normal': surf_normal,
    })

    return rets




def depths_to_points(world_view_transform, image_width, image_height, FoVx, FoVy, depthmap):
    c2w = (world_view_transform.T).inverse()
    W, H = image_width, image_height
    fx = W / (2 * math.tan(FoVx / 2.))
    fy = H / (2 * math.tan(FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(world_view_transform, image_width, image_height, FoVx, FoVy, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(world_view_transform, image_width, image_height, FoVx, FoVy, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output





def render_predicted_more_v2_gof(pc : dict, bs,
                                 world_view_transform,
                                 full_proj_transform,
                                 camera_center,
                                 bg_color : torch.Tensor, 
                                 cfg,
                                 kernel_size = 0.0, 
                                 scaling_modifier = 1.0, 
                                 override_color = None, 
                                 subpixel_offset = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc["xyz"][bs], dtype=pc["xyz"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # if focals_pixels == None:
    tanfovx = math.tan(cfg['model']['fov'] * np.pi / 360)
    tanfovy = math.tan(cfg['model']['fov'] * np.pi / 360)
    FovX = cfg['model']['fov'] * np.pi / 180
    FovY = cfg['model']['fov'] * np.pi / 180
        # print('tanfovx in render init', tanfovx, tanfovy)
    # else:
    #     tanfovx = math.tan(0.5 * focal2fov(focals_pixels[0].item(), cfg['model']['training_resolution']))
    #     tanfovy = math.tan(0.5 * focal2fov(focals_pixels[1].item(), cfg['model']['training_resolution']))

    image_height=int(cfg['model']['training_resolution'])
    image_width=int(cfg['model']['training_resolution'])
    # if subpixel_offset is None:
    subpixel_offset = torch.zeros((int(image_height), int(image_width), 2), dtype=torch.float32, device="cuda")
        
    raster_settings = GaussianRasterizationSettings_GOF(
        image_height=int(cfg['model']['training_resolution']),
        image_width=int(cfg['model']['training_resolution']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=cfg['model']['max_sh_degree'], # 3 by default
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer_GOF(raster_settings=raster_settings)

    means3D = pc["xyz"][bs] #pc.get_xyz
    means2D = screenspace_points
    opacity = pc["opacity"][bs] # pc.get_opacity_with_3D_filter: warning

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
        # cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    scales = pc["scaling"][bs] #pc.get_scaling_with_3D_filter
    rotations = pc["rotation"][bs] #pc.get_rotation

    view2gaussian_precomp = None
    # pipe.compute_view2gaussian_python = True
    # if pipe.compute_view2gaussian_python: # False by default
        # view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # if pipe.convert_SHs_python: # False by default
        #     shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        #     dir_pp = (pc.get_xyz - camera_center.repeat(pc.get_features.shape[0], 1))
        #     # # we local direction
        #     # cam_pos_local = view2gaussian_precomp[:, 3, :3]
        #     # cam_pos_local_scaled = cam_pos_local / scales
        #     # dir_pp = -cam_pos_local_scaled
        #     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        #     sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # else:
            # shs = pc.get_features
        shs = torch.cat([pc["features_dc"][bs], pc["features_rest"][bs]], dim=1).contiguous()

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            view2gaussian_precomp=view2gaussian_precomp)
    else:
        # colors_precomp = override_color
        colors_precomp = pc["rgbs"][bs]
        
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            view2gaussian_precomp=view2gaussian_precomp)

    # print('rendered_image: ', rendered_image.shape, rendered_image.dtype, rendered_image.min(), rendered_image.max())
    
    # rendered normal -> rendered normal world
    render_normal = rendered_image[3:6,:,:]
    render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
    # print('world_view_transform: ', world_view_transform.shape, world_view_transform.dtype, world_view_transform.min(), world_view_transform.max())
    # c2w = (world_view_transform.T).inverse()
    c2w = (world_view_transform.squeeze().T).inverse()
    normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
    render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
    # rendered depth -> depth normal
    # depth_normal, _ = depth_to_normal(world_view_transform, image_width, image_height, FovX, FovY, rendered_image[6:7,:,:])
    depth_normal = depth_to_normal(world_view_transform.squeeze(), image_width, image_height, FovX, FovY, rendered_image[6:7,:,:])
    depth_normal = depth_normal.permute(2, 0, 1) # 3,h,w
    # print('depth_normal: ', depth_normal.shape, depth_normal.dtype, depth_normal.min(), depth_normal.max())
    # print('rendered_image[7:8,:,:]: ', rendered_image[7:8,:,:].shape, rendered_image[7:8,:,:].dtype, rendered_image[7:8,:,:].min(), rendered_image[7:8,:,:].max())

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image[:3,:,:],
            "rendered_normal": render_normal_world,
            "rendered_depth": rendered_image[6:7,:,:],
            "depth_normal": depth_normal,
            "rendered_alpha": rendered_image[7:8,:,:],
            "distortion_map": rendered_image[8:9,:,:],
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def render_predicted_more_v2_gof_in(points3D, pc : dict, bs,
                                 world_view_transform,
                                 full_proj_transform,
                                 camera_center,
                                 bg_color : torch.Tensor, 
                                 cfg,
                                 kernel_size = 0.0, 
                                 scaling_modifier = 1.0, 
                                 override_color = None, 
                                 subpixel_offset = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc["xyz"][bs], dtype=pc["xyz"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # if focals_pixels == None:
    tanfovx = math.tan(cfg['model']['fov'] * np.pi / 360)
    tanfovy = math.tan(cfg['model']['fov'] * np.pi / 360)
    FovX = cfg['model']['fov'] * np.pi / 180
    FovY = cfg['model']['fov'] * np.pi / 180
        # print('tanfovx in render init', tanfovx, tanfovy)
    # else:
    #     tanfovx = math.tan(0.5 * focal2fov(focals_pixels[0].item(), cfg['model']['training_resolution']))
    #     tanfovy = math.tan(0.5 * focal2fov(focals_pixels[1].item(), cfg['model']['training_resolution']))

    image_height=int(cfg['model']['training_resolution'])
    image_width=int(cfg['model']['training_resolution'])
    # if subpixel_offset is None:
    subpixel_offset = torch.zeros((int(image_height), int(image_width), 2), dtype=torch.float32, device="cuda")
        
    raster_settings = GaussianRasterizationSettings_GOF(
        image_height=int(cfg['model']['training_resolution']),
        image_width=int(cfg['model']['training_resolution']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=cfg['model']['max_sh_degree'], # 3 by default
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer_GOF(raster_settings=raster_settings)

    means3D = pc["xyz"][bs] #pc.get_xyz
    means2D = screenspace_points
    opacity = pc["opacity"][bs] # pc.get_opacity_with_3D_filter: warning

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
        # cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    scales = pc["scaling"][bs] #pc.get_scaling_with_3D_filter
    rotations = pc["rotation"][bs] #pc.get_rotation

    view2gaussian_precomp = None
    # pipe.compute_view2gaussian_python = True
    # if pipe.compute_view2gaussian_python: # False by default
        # view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # if pipe.convert_SHs_python: # False by default
        #     shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        #     dir_pp = (pc.get_xyz - camera_center.repeat(pc.get_features.shape[0], 1))
        #     # # we local direction
        #     # cam_pos_local = view2gaussian_precomp[:, 3, :3]
        #     # cam_pos_local_scaled = cam_pos_local / scales
        #     # dir_pp = -cam_pos_local_scaled
        #     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        #     sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # else:
            # shs = pc.get_features
        shs = torch.cat([pc["features_dc"][bs], pc["features_rest"][bs]], dim=1).contiguous()

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, alpha_integrated, color_integrated, radii = rasterizer.integrate(
            points3D = points3D,
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            view2gaussian_precomp=view2gaussian_precomp)
    else:
        # colors_precomp = override_color
        colors_precomp = pc["rgbs"][bs]
        
        rendered_image, alpha_integrated, color_integrated, radii = rasterizer.integrate(
            points3D = points3D,
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            view2gaussian_precomp=view2gaussian_precomp)

    # print('rendered_image: ', rendered_image.shape, rendered_image.dtype, rendered_image.min(), rendered_image.max())
    
    # rendered normal -> rendered normal world
    render_normal = rendered_image[3:6,:,:]
    render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
    # print('world_view_transform: ', world_view_transform.shape, world_view_transform.dtype, world_view_transform.min(), world_view_transform.max())
    # c2w = (world_view_transform.T).inverse()
    c2w = (world_view_transform.squeeze().T).inverse()
    normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
    render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
    # rendered depth -> depth normal
    # depth_normal, _ = depth_to_normal(world_view_transform, image_width, image_height, FovX, FovY, rendered_image[6:7,:,:])
    depth_normal = depth_to_normal(world_view_transform.squeeze(), image_width, image_height, FovX, FovY, rendered_image[6:7,:,:])
    depth_normal = depth_normal.permute(2, 0, 1) # 3,h,w
    # print('depth_normal: ', depth_normal.shape, depth_normal.dtype, depth_normal.min(), depth_normal.max())
    # print('rendered_image[7:8,:,:]: ', rendered_image[7:8,:,:].shape, rendered_image[7:8,:,:].dtype, rendered_image[7:8,:,:].min(), rendered_image[7:8,:,:].max())

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image[:3,:,:],
            "rendered_normal": render_normal_world,
            "rendered_depth": rendered_image[6:7,:,:],
            "depth_normal": depth_normal,
            "rendered_alpha": rendered_image[7:8,:,:],
            "distortion_map": rendered_image[8:9,:,:],
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,

            "alpha_integrated": alpha_integrated,
            "color_integrated": color_integrated,

            "radii": radii}



def render_predicted_more_v3_gof(pc : dict, bs,
                                 world_view_transform,
                                 full_proj_transform,
                                 camera_center,
                                 bg_color : torch.Tensor, 
                                 cfg,
                                 kernel_size = 0.0, 
                                 scaling_modifier = 1.0, 
                                 override_color = None, 
                                 subpixel_offset = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc[bs]["xyz"], dtype=pc[bs]["xyz"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # if focals_pixels == None:
    tanfovx = math.tan(cfg['model']['fov'] * np.pi / 360)
    tanfovy = math.tan(cfg['model']['fov'] * np.pi / 360)
    FovX = cfg['model']['fov'] * np.pi / 180
    FovY = cfg['model']['fov'] * np.pi / 180
        # print('tanfovx in render init', tanfovx, tanfovy)
    # else:
    #     tanfovx = math.tan(0.5 * focal2fov(focals_pixels[0].item(), cfg['model']['training_resolution']))
    #     tanfovy = math.tan(0.5 * focal2fov(focals_pixels[1].item(), cfg['model']['training_resolution']))

    image_height=int(cfg['model']['training_resolution'])
    image_width=int(cfg['model']['training_resolution'])
    # if subpixel_offset is None:
    subpixel_offset = torch.zeros((int(image_height), int(image_width), 2), dtype=torch.float32, device="cuda")
        
    raster_settings = GaussianRasterizationSettings_GOF(
        image_height=int(cfg['model']['training_resolution']),
        image_width=int(cfg['model']['training_resolution']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=cfg['model']['max_sh_degree'], # 3 by default
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer_GOF(raster_settings=raster_settings)

    means3D = pc[bs]["xyz"] #pc.get_xyz
    means2D = screenspace_points
    opacity = pc[bs]["opacity"] # pc.get_opacity_with_3D_filter: warning

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
        # cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    scales = pc[bs]["scaling"] #pc.get_scaling_with_3D_filter
    rotations = pc[bs]["rotation"] #pc.get_rotation

    view2gaussian_precomp = None
    # pipe.compute_view2gaussian_python = True
    # if pipe.compute_view2gaussian_python: # False by default
        # view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # if pipe.convert_SHs_python: # False by default
        #     shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        #     dir_pp = (pc.get_xyz - camera_center.repeat(pc.get_features.shape[0], 1))
        #     # # we local direction
        #     # cam_pos_local = view2gaussian_precomp[:, 3, :3]
        #     # cam_pos_local_scaled = cam_pos_local / scales
        #     # dir_pp = -cam_pos_local_scaled
        #     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        #     sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # else:
            # shs = pc.get_features
        shs = torch.cat([pc[bs]["features_dc"], pc[bs]["features_rest"]], dim=1).contiguous()

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=view2gaussian_precomp)
    else:
        # colors_precomp = override_color
        colors_precomp = pc[bs]["rgbs"]

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=view2gaussian_precomp)


    
    # rendered normal -> rendered normal world
    render_normal = rendered_image[3:6,:,:]
    render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
    c2w = (world_view_transform.squeeze().T).inverse()
    normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
    render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
    # rendered depth -> depth normal
    depth_normal = depth_to_normal(world_view_transform.squeeze(), image_width, image_height, FovX, FovY, rendered_image[6:7,:,:])
    depth_normal = depth_normal.permute(2, 0, 1) # 3,h,w

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image[:3,:,:],
            "rendered_normal": render_normal_world,
            "rendered_depth": rendered_image[6:7,:,:],
            "depth_normal": depth_normal,
            "rendered_alpha": rendered_image[7:8,:,:],
            "distortion_map": rendered_image[8:9,:,:],
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}




def render_predicted_more_v2_gof_fov(pc : dict, bs,
                                 world_view_transform,
                                 full_proj_transform,
                                 camera_center,
                                 bg_color : torch.Tensor, 
                                 cfg,
                                 kernel_size = 0.0, 
                                 scaling_modifier = 1.0, 
                                 override_color = None, 
                                 subpixel_offset = None,
                                 focal_new = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc["xyz"][bs], dtype=pc["xyz"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # if focals_pixels == None:
    # tanfovx = math.tan(cfg['model']['fov'] * np.pi / 360)
    # tanfovy = math.tan(cfg['model']['fov'] * np.pi / 360)
    # FovX = cfg['model']['fov'] * np.pi / 180
    # FovY = cfg['model']['fov'] * np.pi / 180
    # print('focal_new in render init: ', focal_new.shape, focal_new.dtype, focal_new.min(), focal_new.max(), focal_new.requires_grad)

    fov_x_new = focal2fov_torch(focal_new[0,0], cfg['model']['training_resolution'])
    fov_y_new = focal2fov_torch(focal_new[0,1], cfg['model']['training_resolution'])
    # tanfovx = math.tan(fov_x_new * np.pi / 360)
    # tanfovy = math.tan(fov_y_new * np.pi / 360)
    # FovX = fov_x_new * np.pi / 180
    # FovY = fov_y_new * np.pi / 180
    tanfovx = math.tan(fov_x_new / 2)
    tanfovy = math.tan(fov_y_new / 2)
    FovX = fov_x_new
    FovY = fov_y_new

        # print('tanfovx in render init', tanfovx, tanfovy)
    # else:
    #     tanfovx = math.tan(0.5 * focal2fov(focals_pixels[0].item(), cfg['model']['training_resolution']))
    #     tanfovy = math.tan(0.5 * focal2fov(focals_pixels[1].item(), cfg['model']['training_resolution']))

    image_height=int(cfg['model']['training_resolution'])
    image_width=int(cfg['model']['training_resolution'])
    # if subpixel_offset is None:
    subpixel_offset = torch.zeros((int(image_height), int(image_width), 2), dtype=torch.float32, device="cuda")
        
    raster_settings = GaussianRasterizationSettings_GOF_FOV(
        image_height=int(cfg['model']['training_resolution']),
        image_width=int(cfg['model']['training_resolution']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=cfg['model']['max_sh_degree'], # 3 by default
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer_GOF_FOV(raster_settings=raster_settings)

    means3D = pc["xyz"][bs] #pc.get_xyz
    means2D = screenspace_points
    opacity = pc["opacity"][bs] # pc.get_opacity_with_3D_filter: warning

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
        # cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    scales = pc["scaling"][bs] #pc.get_scaling_with_3D_filter
    rotations = pc["rotation"][bs] #pc.get_rotation

    view2gaussian_precomp = None
    # pipe.compute_view2gaussian_python = True
    # if pipe.compute_view2gaussian_python: # False by default
        # view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # if pipe.convert_SHs_python: # False by default
        #     shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        #     dir_pp = (pc.get_xyz - camera_center.repeat(pc.get_features.shape[0], 1))
        #     # # we local direction
        #     # cam_pos_local = view2gaussian_precomp[:, 3, :3]
        #     # cam_pos_local_scaled = cam_pos_local / scales
        #     # dir_pp = -cam_pos_local_scaled
        #     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        #     sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # else:
            # shs = pc.get_features
        shs = torch.cat([pc["features_dc"][bs], pc["features_rest"][bs]], dim=1).contiguous()
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=view2gaussian_precomp,
        focal_new=focal_new)
    
    # # rendered normal -> rendered normal world
    # render_normal = rendered_image[3:6,:,:]
    # render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
    # # print('world_view_transform: ', world_view_transform.shape, world_view_transform.dtype, world_view_transform.min(), world_view_transform.max())
    # # c2w = (world_view_transform.T).inverse()
    # c2w = (world_view_transform.squeeze().T).inverse()
    # normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
    # render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
    # # rendered depth -> depth normal
    # # depth_normal, _ = depth_to_normal(world_view_transform, image_width, image_height, FovX, FovY, rendered_image[6:7,:,:])
    # depth_normal = depth_to_normal(world_view_transform.squeeze(), image_width, image_height, FovX, FovY, rendered_image[6:7,:,:])
    # depth_normal = depth_normal.permute(2, 0, 1) # 3,h,w
    # # print('depth_normal: ', depth_normal.shape, depth_normal.dtype, depth_normal.min(), depth_normal.max())
    # # print('rendered_image[7:8,:,:]: ', rendered_image[7:8,:,:].shape, rendered_image[7:8,:,:].dtype, rendered_image[7:8,:,:].min(), rendered_image[7:8,:,:].max())

    # # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image[:3,:,:],
            # "rendered_normal": render_normal_world,
            "rendered_depth": rendered_image[6:7,:,:],
            # "depth_normal": depth_normal,
            "rendered_alpha": rendered_image[7:8,:,:],
            # "distortion_map": rendered_image[8:9,:,:],
            # "viewspace_points": screenspace_points,
            # "visibility_filter" : radii > 0,
            # "radii": radii
            }

def render_predicted_more_v3_gof_fov(pc : dict, bs,
                                 world_view_transform,
                                 full_proj_transform,
                                 camera_center,
                                 bg_color : torch.Tensor, 
                                 cfg,
                                 kernel_size = 0.0, 
                                 scaling_modifier = 1.0, 
                                 override_color = None, 
                                 subpixel_offset = None,
                                 focal_new=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc[bs]["xyz"], dtype=pc[bs]["xyz"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # if focals_pixels == None:
    # tanfovx = math.tan(cfg['model']['fov'] * np.pi / 360)
    # tanfovy = math.tan(cfg['model']['fov'] * np.pi / 360)
    # FovX = cfg['model']['fov'] * np.pi / 180
    # FovY = cfg['model']['fov'] * np.pi / 180

    fov_x_new = focal2fov_torch(focal_new[0,0], cfg['model']['training_resolution'])
    fov_y_new = focal2fov_torch(focal_new[0,1], cfg['model']['training_resolution'])
    # tanfovx = math.tan(fov_x_new * np.pi / 360)
    # tanfovy = math.tan(fov_y_new * np.pi / 360)
    # FovX = fov_x_new * np.pi / 180
    # FovY = fov_y_new * np.pi / 180
    tanfovx = math.tan(fov_x_new / 2)
    tanfovy = math.tan(fov_y_new / 2)
    FovX = fov_x_new
    FovY = fov_y_new
        # print('tanfovx in render init', tanfovx, tanfovy)
    # else:
    #     tanfovx = math.tan(0.5 * focal2fov(focals_pixels[0].item(), cfg['model']['training_resolution']))
    #     tanfovy = math.tan(0.5 * focal2fov(focals_pixels[1].item(), cfg['model']['training_resolution']))

    image_height=int(cfg['model']['training_resolution'])
    image_width=int(cfg['model']['training_resolution'])
    # if subpixel_offset is None:
    subpixel_offset = torch.zeros((int(image_height), int(image_width), 2), dtype=torch.float32, device="cuda")
        
    raster_settings = GaussianRasterizationSettings_GOF_FOV(
        image_height=int(cfg['model']['training_resolution']),
        image_width=int(cfg['model']['training_resolution']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=cfg['model']['max_sh_degree'], # 3 by default
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer_GOF_FOV(raster_settings=raster_settings)

    means3D = pc[bs]["xyz"] #pc.get_xyz
    means2D = screenspace_points
    opacity = pc[bs]["opacity"] # pc.get_opacity_with_3D_filter: warning

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
        # cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    scales = pc[bs]["scaling"] #pc.get_scaling_with_3D_filter
    rotations = pc[bs]["rotation"] #pc.get_rotation

    view2gaussian_precomp = None
    # pipe.compute_view2gaussian_python = True
    # if pipe.compute_view2gaussian_python: # False by default
        # view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # if pipe.convert_SHs_python: # False by default
        #     shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        #     dir_pp = (pc.get_xyz - camera_center.repeat(pc.get_features.shape[0], 1))
        #     # # we local direction
        #     # cam_pos_local = view2gaussian_precomp[:, 3, :3]
        #     # cam_pos_local_scaled = cam_pos_local / scales
        #     # dir_pp = -cam_pos_local_scaled
        #     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        #     sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # else:
            # shs = pc.get_features
        shs = torch.cat([pc[bs]["features_dc"], pc[bs]["features_rest"]], dim=1).contiguous()
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=view2gaussian_precomp,
        focal_new=focal_new)
    
    # # rendered normal -> rendered normal world
    # render_normal = rendered_image[3:6,:,:]
    # render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
    # c2w = (world_view_transform.squeeze().T).inverse()
    # normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
    # render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
    # # rendered depth -> depth normal
    # depth_normal = depth_to_normal(world_view_transform.squeeze(), image_width, image_height, FovX, FovY, rendered_image[6:7,:,:])
    # depth_normal = depth_normal.permute(2, 0, 1) # 3,h,w

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image[:3,:,:],
            # "rendered_normal": render_normal_world,
            "rendered_depth": rendered_image[6:7,:,:],
            # "depth_normal": depth_normal,
            "rendered_alpha": rendered_image[7:8,:,:],
            # "distortion_map": rendered_image[8:9,:,:],
            # "viewspace_points": screenspace_points,
            # "visibility_filter" : radii > 0,
            # "radii": radii
            }

