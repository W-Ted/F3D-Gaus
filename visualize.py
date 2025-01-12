import argparse
import yaml
import os
from tqdm import tqdm, trange

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import math

from src import dataset_dict
from src.camera import compute_cam2world_matrix
from src.utils import TensorGroup, count_parameters, colorize, sample_front_circle_gs
from src.utils import TensorGroup, colorize_first

import imageio
import numpy as np
from torchvision.utils import make_grid

from src.unet_gs import Unet_GS_gtunet
from src.gaussian_renderer import render_predicted_more_v2_gof, render_predicted_more_v2_gof_in
from src.dataio_gs_test_256_demo import matrix_to_quaternion, update_camera_pose
from plyfile import PlyData, PlyElement
from typing import Tuple, Optional
from collections import OrderedDict
import cv2

import trimesh
from einops import einsum
from tetranerf.utils.extension import cpp
from src.utils_tetmesh import marching_tetrahedra

STYLE = 'lod_no' # Vanilla

torch.manual_seed(0)
np.random.seed(0)

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))
    # return pixels / (2 * abs(math.tan(fov / 2)))

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def get_scaling_with_3D_filter(scale):
    scales = scale    
    scales = torch.square(scales) + 0 #torch.square(self.filter_3D)
    scales = torch.sqrt(scales)
    return scales

@torch.no_grad()
def get_frustum_mask(points: torch.Tensor, cameras, near: float = 0.02, far: float = 1e6, fov=60):
    # H, W = cameras[0].image_height, cameras[0].image_width
    H, W = 256, 256
    focal_x = fov2focal(fov, 256)
    focal_y = fov2focal(fov, 256)

    intrinsics = torch.stack(
        [
            torch.Tensor(
                [[focal_x, 0, W / 2],
                 [0, focal_y, H / 2],
                 [0, 0, 1]]
            ) for i in range(cameras.shape[0])
        ], 
        dim=0
    ).to(points.device)

    # full_proj_matrices: (n_view, 4, 4)
    # view_matrices = torch.stack(
    #     [cameras[i] for i in range(cameras.shape[0])], dim=0
    # ).transpose(1, 2)
    view_matrices = cameras.squeeze().transpose(1, 2) # ?

    ones = torch.ones_like(points[:, 0]).unsqueeze(-1)
    # homo_points: (N, 4)
    homo_points = torch.cat([points, ones], dim=-1)

    # uv_points: (n_view, N, 4, 4)
    # Apply batch matrix multiplication to get uv_points for all cameras
    view_points = einsum(view_matrices, homo_points, "n_view b c, N c -> n_view N b")
    view_points = view_points[:, :, :3]

    uv_points = einsum(intrinsics, view_points, "n_view b c, n_view N c -> n_view N b")

    z = uv_points[:, :, -1:]
    uv_points = uv_points[:, :, :2] / z
    u, v = uv_points[:, :, 0], uv_points[:, :, 1]

    # Optionally, we can apply near-far culling
    # Apply near-far culling
    depth = view_points[:, :, -1]
    cull_near_fars = (depth >= near) & (depth <= far)

    # Apply frustum mask
    mask = torch.any(cull_near_fars & (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1), dim=0)
    return mask

@torch.no_grad()
def get_tetra_points(views, near, far, fov, rotation, xyz, scale):
    M = trimesh.creation.box()
    M.vertices *= 2
    
    rots = build_rotation(rotation)
    xyz = xyz
    scale = get_scaling_with_3D_filter(scale) * 3. # TODO test
    
    vertices = M.vertices.T    
    vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
    # scale vertices first
    vertices = vertices * scale.unsqueeze(-1)
    vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
    vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
    # concat center points
    vertices = torch.cat([vertices, xyz], dim=0)
    
    # scale is not a good solution but use it for now
    scale = scale.max(dim=-1, keepdim=True)[0]
    scale_corner = scale.repeat(1, 8).reshape(-1, 1)
    vertices_scale = torch.cat([scale_corner, scale], dim=0)
    
    # Mask out vertices outside of context views
    vertex_mask = get_frustum_mask(vertices, views, near, far, fov)
    return vertices[vertex_mask], vertices_scale[vertex_mask]

def load_ply(gs_dic, bb, path):
    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    xyz = gs_dic['xyz'][bb].detach() #.cpu().numpy()
    normals = torch.zeros_like(xyz)
    f_dc = gs_dic['features_dc'][bb].detach().transpose(1, 2).flatten(start_dim=1).contiguous() #.cpu().numpy()
    f_rest = torch.zeros_like(gs_dic['features_dc'][bb])
    f_rest = f_rest.expand([-1,(3+1)**2-1,-1]).detach().transpose(1, 2).flatten(start_dim=1).contiguous() #.cpu().numpy()

    opacities = gs_dic['opacity'][bb].detach() #.cpu().numpy()
    scale = gs_dic['scaling'][bb].detach() #.cpu().numpy()
    rotation = gs_dic['rotation'][bb].detach() #.cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

    if path is not None:
        PlyData([el]).write(path)
    else:
        return xyz, f_dc, f_rest, opacities, scale, rotation


def main(config, args):
    # define params
    image_size = config['model']['training_resolution']
    bs =  config['F3D-Gaus']['training']['batch_size']

    # init dataloader
    if config['F3D-Gaus']['training']['dataset'].endswith('debug') or config['opt']['depth_type']=='leres' or config['opt']['depth_type']=='marigold' or config['opt']['depth_type']=='da2':
        dataset_class = dataset_dict[config['F3D-Gaus']['training']['dataset']+'_demo']
    else:
        dataset_class = dataset_dict[config['F3D-Gaus']['training']['dataset']+'_debug']
    dataset = dataset_class(args.folder, image_size=image_size, config=config, random_flip=False)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False, num_workers=8)

    # init model
    model = Unet_GS_gtunet(
        cfg=config,
        renderer = None
    )

    # load pre-trained model
    if args.load_model!='default.pt':
        load_model = args.load_model
        checkpoint = torch.load(load_model, map_location='cpu')
        if 'module' in list(checkpoint['model'].keys())[0]:
            state_dict = {k.partition('module.')[2]: checkpoint['model'][k] for k in checkpoint['model'].keys()}
        else:
            state_dict = dict(checkpoint['model'])
            print(state_dict.keys())
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    else:
        raise Exception("Please provide pretrained model file")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataloader_iter = iter(dataloader)
    step = 0

    # for each sample
    for data in  dataloader_iter:
        with torch.no_grad():
            step += 1
            bs = data['images'].shape[0] # 8

            x_start = data['images']
            input_feat = x_start.unsqueeze(1).to(device)
            background = torch.tensor([0,0,0], dtype=torch.float).to(device)
            background = background.unsqueeze(0).expand([bs, -1]).contiguous() # [bs,3]

            fov_diff = 0.0
            yaw_diff = 0.25
            pitch_diff = 0.15

            # first-forward 
            num_frames=8 # number of views
            filename=step
            output_path=args.output_path + f'_yaw{yaw_diff}_pitch{pitch_diff}_aug{num_frames}_demo'

            # define spiral c2w
            camera_params = TensorGroup(
                angles=torch.zeros(1,3),
                fov=torch.ones(1)*config['model']['fov'], 
                radius=torch.ones(1)*config['model']['radius'],
                look_at=torch.zeros(1,3),
            )
            camera_params.look_at[:,2] = config['model']['look_at']
            camera_samples = sample_front_circle_gs(camera_params, num_frames, fov_diff=fov_diff, yaw_diff=yaw_diff, pitch_diff=pitch_diff)
            cam2w = compute_cam2world_matrix(camera_samples).to(device)
            cam2w = torch.inverse(cam2w)

            # prepare camera params
            Rt = torch.inverse(cam2w).contiguous() # [bs, 4, 4]
            world_view_transforms = Rt.transpose(1,2).unsqueeze(1).contiguous() # [bs, 1, 4, 4]
            view_to_world_transforms = cam2w.transpose(1,2).unsqueeze(1).contiguous() # [bs, 1, 4, 4]
            camera_centers = world_view_transforms.inverse()[:, :, 3, :3].contiguous() # [bs, 1, 3]
            projection_matrix = dataset.projection_matrix.expand([cam2w.shape[0], -1, -1]).unsqueeze(1).contiguous().to(device)
            full_proj_transforms = (world_view_transforms[:,0,:,:].bmm(projection_matrix[:,0,:,:])).unsqueeze(1).contiguous() # [bs, 1, 4, 4]

            if config['opt']['update_pose']:
                    world_view_transforms, \
                    view_to_world_transforms, \
                    full_proj_transforms, \
                    camera_centers, \
                    _ \
                    = update_camera_pose(
                        world_view_transforms, \
                        view_to_world_transforms, \
                        full_proj_transforms, \
                        camera_centers,
                        dataset.inverse_first_camera.to(device),
                        False
                    )
            
            source_cv2wT_quat = torch.zeros_like(full_proj_transforms[:,0:1,0,:]) # [bs, 1, 4]
            for i in range(num_frames):     
                source_cv2wT_quat[i] = matrix_to_quaternion(view_to_world_transforms[i,0,:3,:3].transpose(0,1).contiguous())
            cano_view_to_world_transforms = dataset.view_to_world_transforms.expand([bs, -1, -1]).unsqueeze(1).contiguous().to(device)
            cano_source_cv2wT_quat = dataset.source_cv2wT_quat.expand([bs, -1, -1]).contiguous().to(device)

            # F3D-Gaus forward
            input_feat = torch.cat([input_feat, torch.ones_like(input_feat[:,:,0:1,:,:])], 2)
            _, _, gaussian_splat_batch = model(input_feat, background, cano_view_to_world_transforms, cano_source_cv2wT_quat, return_3d_features=True, render=False, squre_clip=config['opt']['squre_clip'], unet_depth=data['depth'].to(device))
            
            # pseudo depth (for visulization)
            unet_depth = gaussian_splat_batch['unet_depth'].reshape(x_start.shape[0], -1, x_start.shape[-2], x_start.shape[-1]) # bnc -> bchw

            # rendering of 8 views
            frames = []
            rendered_8 = []
            alpha_8 = []
            depth_8 = []
            for th in tqdm(range(num_frames)): # 8 views for aggregation
                rgb_reshape = []
                depth = []
                alpha = []
                for bb in range(bs):
                    output_dic = render_predicted_more_v2_gof(gaussian_splat_batch, bb,
                                    world_view_transforms[th:th+1].contiguous(),
                                    full_proj_transforms[th:th+1].contiguous(),
                                    camera_centers[th:th+1].contiguous(),
                                    background[th:th+1].contiguous(),
                                    config)
                    rgb_reshape += [output_dic["render"].reshape(-1,3,image_size,image_size).cpu()] #[0-1]
                    depth += [output_dic["rendered_depth"].reshape(-1,1,image_size, image_size).cpu()]
                    alpha += [output_dic["rendered_alpha"].reshape(-1,1,image_size, image_size).cpu()]
                rgb_reshape = torch.concat(rgb_reshape, 0)
                depth = torch.concat(depth, 0)
                alpha = torch.concat(alpha, 0)

                rgb_reshape = rgb_reshape.clamp(0, 1)
                rendered_8.append(rgb_reshape)
                alpha_8.append(alpha)
                depth_8.append(depth)

            alpha_8 = torch.stack(alpha_8) 
            alpha_8 = alpha_8.transpose(0,1) 

            depth_8 = torch.stack(depth_8) 
            depth_8 = depth_8.transpose(0,1) 

            rendered_8 = torch.stack(rendered_8) 
            rendered_8 = rendered_8.transpose(0,1) 

            # merge 8 3DGSs to get the final 3DGS representation
            gaussian_splat_batch_merge = {}
            for aux_i in range(num_frames): # 128
                novel_img = rendered_8[:,aux_i:aux_i+1].to(input_feat.device) # bs 1 3 h w
                novel_view_to_world_transforms = view_to_world_transforms[aux_i:aux_i+1].expand([bs,-1,-1,-1])
                novel_source_cv2wT_quat = source_cv2wT_quat[aux_i:aux_i+1].expand([bs,-1,-1])
                novel_alpha = alpha_8[:,aux_i:aux_i+1].to(input_feat.device)
                novel_img = torch.cat([novel_img, novel_alpha], 2)
                input_novel_d = depth_8[:,aux_i].to(input_feat.device) # bs 1 h w
                _, _, gaussian_splat_batch_i = model(novel_img, background, novel_view_to_world_transforms, novel_source_cv2wT_quat, return_3d_features=True, render=False, squre_clip=config['opt']['squre_clip'], unet_depth=input_novel_d)
                
                for key in gaussian_splat_batch.keys():
                    if aux_i == 0:
                        gaussian_splat_batch_merge[key] = torch.cat([gaussian_splat_batch[key], gaussian_splat_batch_i[key]], 1)
                    else:
                        gaussian_splat_batch_merge[key] = torch.cat([gaussian_splat_batch_merge[key], gaussian_splat_batch_i[key]], 1)
            
            # re-define rendering views
            num_frames=128
            camera_samples = sample_front_circle_gs(camera_params, num_frames, fov_diff=fov_diff, yaw_diff=yaw_diff, pitch_diff=pitch_diff)
            cam2w = compute_cam2world_matrix(camera_samples).to(device)
            cam2w = torch.inverse(cam2w)

            # add one more frontal view
            yaw_diff = 0.0
            pitch_diff = 0.0
            num_frames_f=1
            camera_samples_f = sample_front_circle_gs(camera_params, num_frames_f, fov_diff=fov_diff, yaw_diff=yaw_diff, pitch_diff=pitch_diff)
            cam2w_f = compute_cam2world_matrix(camera_samples_f).to(device)
            cam2w_f = torch.inverse(cam2w_f)
            cam2w = torch.cat([cam2w_f, cam2w],0)

            # prepare params
            Rt = torch.inverse(cam2w).contiguous() # [bs, 4, 4]
            world_view_transforms = Rt.transpose(1,2).unsqueeze(1).contiguous() # [bs, 1, 4, 4]
            view_to_world_transforms = cam2w.transpose(1,2).unsqueeze(1).contiguous() # [bs, 1, 4, 4]
            camera_centers = world_view_transforms.inverse()[:, :, 3, :3].contiguous() # [bs, 1, 3]
            projection_matrix = dataset.projection_matrix.expand([cam2w.shape[0], -1, -1]).unsqueeze(1).contiguous().to(device)
            full_proj_transforms = (world_view_transforms[:,0,:,:].bmm(projection_matrix[:,0,:,:])).unsqueeze(1).contiguous() # [bs, 1, 4, 4]
            source_cv2wT_quat = torch.zeros_like(full_proj_transforms[:,0:1,0,:]) # [bs, 1, 4]
            for i in range(num_frames):     
                source_cv2wT_quat[i] = matrix_to_quaternion(view_to_world_transforms[i,0,:3,:3].transpose(0,1).contiguous())

            if config['opt']['update_pose']:
                    world_view_transforms, \
                    view_to_world_transforms, \
                    full_proj_transforms, \
                    camera_centers, \
                    _ \
                    = update_camera_pose(
                        world_view_transforms, \
                        view_to_world_transforms, \
                        full_proj_transforms, \
                        camera_centers,
                        dataset.inverse_first_camera.to(device),
                        False
                    )

            # novel view synthesis
            frames = []
            # print('Visualizing file: ', filename)
            # for th in tqdm(range(num_frames+num_frames_f)):
            for thh in tqdm(range(num_frames)):
                th = thh + 1

                rgb_reshape = []
                depth = []
                depth_normal = []
                for bb in range(bs):
                    output_dic = render_predicted_more_v2_gof(gaussian_splat_batch_merge, bb, 
                                    world_view_transforms[th:th+1].contiguous(),
                                    full_proj_transforms[th:th+1].contiguous(),
                                    camera_centers[th:th+1].contiguous(),
                                    background[th:th+1].contiguous(),
                                    config)
                    rgb_reshape += [output_dic["render"].reshape(-1,3,image_size,image_size).cpu()] #[0-1]
                    depth += [output_dic["rendered_depth"].reshape(-1,1,image_size, image_size).cpu()]
                    depth_normal += [output_dic["depth_normal"].reshape(-1,3,image_size, image_size).cpu()]
                rgb_reshape = torch.concat(rgb_reshape, 0)
                depth = torch.concat(depth, 0)
                depth_normal = torch.concat(depth_normal, 0)

                rgb_reshape = rgb_reshape.clamp(0, 1)
                unet_depth_reshape, vmin, vmax = colorize_first(unet_depth.clone().cpu(), cmap='magma_r') # b 128 128 4
                unet_depth_reshape = torch.from_numpy(unet_depth_reshape).to(rgb_reshape.device).permute(0,3,1,2)[:,:3]/255
                depth_reshape = colorize(depth, cmap='magma_r', vmin=vmin, vmax=vmax) # b 128 128 4
                depth_reshape = torch.from_numpy(depth_reshape).to(rgb_reshape.device).permute(0,3,1,2)[:,:3]/255
                depth_normal_reshape = (depth_normal.to(rgb_reshape.device).expand([-1,3,-1,-1]) + 1) / 2

                combined = torch.cat([x_start, unet_depth_reshape, rgb_reshape, depth_reshape, depth_normal_reshape], dim=3)
                combined = make_grid(combined, nrow = int(math.sqrt(bs)))
                frames.append((255*np.clip(combined.permute(1,2,0).cpu().detach().numpy(), 0, 1)).astype(np.uint8))

            os.makedirs(output_path, exist_ok=True)
            imageio.mimwrite(os.path.join(output_path, f'output-{filename}.mp4'), frames, fps=40, quality=8)


            if not args.skip_mesh:
                # save meshes
                for bb in trange(bs):
                    # extract attributes for each 3DGS
                    if args.aug_mesh:
                        xyz, f_dc, f_rest, opacities, scale, rotation = load_ply(gaussian_splat_batch_merge, bb=bb, path=None) # batch merge
                    else:
                        xyz, f_dc, f_rest, opacities, scale, rotation = load_ply(gaussian_splat_batch, bb=bb, path=None) # batch
                    filter_3D = torch.tensor([0,0,0], dtype=torch.float, device="cuda")
                    
                    bg_color = [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    kernel_size = 0 #dataset.kernel_size
                
                    os.makedirs(os.path.join(output_path, f'{filename:02d}_{bb:02d}'), exist_ok=True)
                    near = config['dataset_params']['z_near']
                    far = config['dataset_params']['z_far']
                    fov = config['model']['fov']
                    points, points_scale = get_tetra_points(world_view_transforms, near, far, fov, rotation, xyz, scale)

                    # print("create cells and save")
                    cells = cpp.triangulate(points)
                    # we should filter the cell if it is larger than the gaussians
                    torch.save(cells, os.path.join(output_path, f'{filename:02d}_{bb:02d}', "cells.pt"))
                    
                    # evaluate alpha
                    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
                    for th in tqdm(range(num_frames_f+num_frames)):
                        # for bb in range(bs):
                            # if bb > 0:
                            # if bb == 0:
                                # continue
                        output_dic = render_predicted_more_v2_gof_in(points, gaussian_splat_batch_merge, bb, 
                                        world_view_transforms[th:th+1].contiguous(),
                                        full_proj_transforms[th:th+1].contiguous(),
                                        camera_centers[th:th+1].contiguous(),
                                        background[th:th+1].contiguous(),
                                        config)
                        alpha_integrated = output_dic["alpha_integrated"]
                        final_alpha = torch.min(final_alpha, alpha_integrated)
                    alpha = 1 - final_alpha # alpha:  torch.Size([589824]) torch.float32
                    vertices = points.cuda()[None]
                    tets = cells.cuda().long()

                    def alpha_to_sdf(alpha):    
                        sdf = alpha - 0.5
                        sdf = sdf[None]
                        return sdf
                    sdf = alpha_to_sdf(alpha)

                    torch.cuda.empty_cache()
                    verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf, points_scale[None])
                    torch.cuda.empty_cache()
                    
                    end_points, end_sdf = verts_list[0]
                    end_scales = scale_list[0]
                    
                    faces=faces_list[0].cpu().numpy()
                    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.
                        
                    left_points = end_points[:, 0, :]
                    right_points = end_points[:, 1, :]
                    left_sdf = end_sdf[:, 0, :]
                    right_sdf = end_sdf[:, 1, :]
                    left_scale = end_scales[:, 0, 0]
                    right_scale = end_scales[:, 1, 0]
                    distance = torch.norm(left_points - right_points, dim=-1) # 0 - 2.26
                    scale = left_scale + right_scale                          # 0 - 0.42

                    n_binary_steps = 8
                    for stepp in range(n_binary_steps):
                        # print("binary search in step {}".format(stepp))
                        mid_points = (left_points + right_points) / 2
                        
                        final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
                        for th in tqdm(range(num_frames_f+num_frames)):
                            output_dic = render_predicted_more_v2_gof_in(points, gaussian_splat_batch_merge, bb, 
                                            world_view_transforms[th:th+1].contiguous(),
                                            full_proj_transforms[th:th+1].contiguous(),
                                            camera_centers[th:th+1].contiguous(),
                                            background[th:th+1].contiguous(),
                                            config)
                            alpha_integrated = output_dic["alpha_integrated"]
                            final_alpha = torch.min(final_alpha, alpha_integrated)
                        alpha = 1 - final_alpha
                        mid_sdf = alpha_to_sdf(alpha).squeeze().unsqueeze(-1)
                        
                        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

                        left_sdf[ind_low] = mid_sdf[ind_low]
                        right_sdf[~ind_low] = mid_sdf[~ind_low]
                        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
                        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
                    
                        points = (left_points + right_points) / 2
                        if stepp not in [7]:
                            continue
                        
                        # if texture_mesh:
                        if False: 
                            final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
                            final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")
                            for th in tqdm(range(num_frames_f+num_frames)):
                                output_dic = render_predicted_more_v2_gof_in(points, gaussian_splat_batch_merge, bb, 
                                                world_view_transforms[th:th+1].contiguous(),
                                                full_proj_transforms[th:th+1].contiguous(),
                                                camera_centers[th:th+1].contiguous(),
                                                background[th:th+1].contiguous(),
                                                config)
                                alpha_integrated = output_dic["alpha_integrated"]
                                color_integrated = output_dic["color_integrated"]
                                final_color = torch.where((alpha_integrated < final_alpha).reshape(-1, 1), color_integrated, final_color)
                            color = final_color
                            vertex_colors=(color.cpu().numpy() * 255).astype(np.uint8)
                        else:
                            vertex_colors=None
                        mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, vertex_colors=vertex_colors, process=False)
                        
                        # filter
                        # if filter_mesh:
                        if True:
                            mask = (distance <= 3 * scale).cpu().numpy() # 0-2.26, 0-0.42
                            face_mask = mask[faces].all(axis=1)
                            mesh.update_vertices(mask)
                            mesh.update_faces(face_mask)
                        
                    mesh.export(os.path.join(output_path, f'{filename:02d}_{bb:02d}', f"mesh_binary_search.ply"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arch parameters')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/arch_parameters_clip.yaml')
    parser.add_argument('--load_model',
                        dest="load_model",
                        help =  'model.pt file local',
                        default='default.pt')
    parser.add_argument('--folder',
                        dest="folder",
                        help =  'folder with rgbd images',
                        default='./images/1/')
    parser.add_argument('--output_path',
                        dest="output_path",
                        help =  'folder to output visualization video',
                        default='./output/')
    parser.add_argument('--skip_mesh',
                        action='store_true',
                        default=False,
                        help =  'skip mesh extraction')
    parser.add_argument('--aug_mesh',
                        action='store_true',
                        default=False,
                        help =  'aggregation for mesh extraction')
    
        

    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    main(config, args)