# https://github.com/snap-research/3dgp

from src.utils import TensorGroup
import torch
from typing import List, Tuple, Dict, Union, Optional
import torch.nn.functional as F

import numpy as np

def normalize(x: torch.Tensor, dim: int=-1) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    output = x / (torch.norm(x, dim=dim, keepdim=True))
    return  output#torch.nan_to_num(output)

def spherical2cartesian(yaw: torch.Tensor, pitch: torch.Tensor, radius: Union[torch.Tensor, float], look_at: torch.Tensor) -> torch.Tensor:
    """
    Converts spherical coordinates to cartesian: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    Rotation must be in [0, 2*pi]
    Pitch must be in [0, pi]
    """
    assert yaw.ndim == pitch.ndim, f"Wrong shapes: {yaw.shape}, {pitch.shape}"
    assert len(yaw) == len(pitch), f"Wrong shapes: {yaw.shape}, {pitch.shape}"

    # These equations reflect our camera conventions. Change with care.
    x = - radius * torch.sin(yaw) * torch.cos(pitch) + look_at[:,0]  # [..., batch_size]
    y = - radius * torch.sin(pitch) + look_at[:,1] # [..., batch_size]
    z = - radius * torch.cos(pitch) * torch.cos(yaw) + look_at[:,2] # [..., batch_size]
    coords = torch.stack([x, y, z], dim=-1) # [..., batch_size, 3]

    return coords



def sample_rays(c2w: torch.Tensor, fov_degrees: torch.Tensor, resolution: Tuple[int, int]):
    batch_size = len(c2w)
    device = c2w.device
    compute_batch_size = batch_size # Batch size used for computations
    w, h = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, w, device=device),
                          torch.linspace(1, -1, h, device=device), indexing='ij')
    x = x.T.flatten().unsqueeze(0).repeat(compute_batch_size, 1) # [compute_batch_size, h * w]
    y = y.T.flatten().unsqueeze(0).repeat(compute_batch_size, 1) # [compute_batch_size, h * w]

    fov_rad = fov_degrees / 360 * 2 * np.pi # [compute_batch_size, 1]
    z = -torch.ones((compute_batch_size, h * w), device=device) / torch.tan(fov_rad * 0.5) # [compute_batch_size, h * w]
    ray_d_cam = normalize(torch.stack([x, y, z], dim=2), dim=2) # [compute_batch_size, h * w, 3]

    if compute_batch_size == 1:
        ray_d_cam = ray_d_cam.repeat(batch_size, 1, 1) # [batch_size, h * w, 3]

    ray_d_world = torch.bmm(c2w[..., :3, :3], ray_d_cam.reshape(batch_size, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(batch_size, h * w, 3) # [batch_size, h * w, 3]
    ray_d_world = torch.nn.functional.normalize(ray_d_world, dim=2)
    homogeneous_origins = torch.zeros((batch_size, 4, h * w), device=device) # [batch_size, 4, h * w]
    homogeneous_origins[:, 3, :] = 1
    ray_o_world = torch.bmm(c2w, homogeneous_origins).permute(0, 2, 1).reshape(batch_size, h * w, 4)[..., :3] # [batch_size, h * w, 3]

    return ray_o_world, ray_d_world



def compute_cam2world_matrix(camera_params: TensorGroup) -> torch.Tensor:
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    camera_params:
        - angles: [batch_size, 3] — yaw/pitch/roll angles
        - radius: [batch_size]
        - look_at: [batch_size, 3] — rotation/elevation/radius of the look-at point.
    """
    origins = spherical2cartesian(camera_params.angles[:, 0], camera_params.angles[:, 1], camera_params.radius, camera_params.look_at) # [batch_size, 3]
    look_at = camera_params.look_at
    # print('origins: ', origins)
    # print('look_at: ', look_at)
    forward_vector = normalize(look_at - origins) # [batch_size, 3]
    # forward_vector = normalize(origins - look_at) # [batch_size, 3]
    batch_size = forward_vector.shape[0]
    forward_vector = normalize(forward_vector) # [batch_size, 3]
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=forward_vector.device).expand_as(forward_vector) # [batch_size, 3]
    left_vector = normalize(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize(torch.cross(forward_vector, left_vector, dim=-1))
    rotation_matrix = torch.eye(4, device=forward_vector.device).unsqueeze(0).repeat(batch_size, 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)
    translation_matrix = torch.eye(4, device=forward_vector.device).unsqueeze(0).repeat(batch_size, 1, 1)
    translation_matrix[:, :3, 3] = origins

    cam2world = translation_matrix @ rotation_matrix

    return cam2world

