import math
import numpy as np

from random import random
from functools import partial
from collections import namedtuple
from src.utils import *
from src.camera import compute_cam2world_matrix, sample_rays, normalize
from omegaconf import OmegaConf

import torch
from torch import nn, einsum
import torch.nn.functional as F

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from torch.utils.checkpoint import checkpoint

from src.gaussian_predictor import GaussianSplatPredictor_gtunet

# from src.model.modules.flow_comp_raft import RAFT_bi
# from src.model.recurrent_flow_completion import RecurrentFlowCompleteNet
# from src.model.propainter import InpaintGenerator



from tqdm import tqdm



class Unet_GS_gtunet(nn.Module):
    def __init__(
        self,
        cfg,
        renderer,
    ):
        super().__init__()

        self.gaussian_predictor = GaussianSplatPredictor_gtunet(cfg)
        # define renderer
        self.renderer = renderer

        self.cfg = cfg
    
    def forward(self, x_input, background, view_to_world_transforms, source_cv2wT_quat, return_3d_features=True, render=False, return_depth=False, squre_clip=10000.0,
                world_view_transforms = None,
                full_proj_transforms = None,
                camera_centers = None,
                config = None,
                image_size = None,
                unet_depth = None
                ):

        if return_depth: # false
            gaussian_splats = self.gaussian_predictor(x_input,
                                             view_to_world_transforms,
                                             source_cv2wT_quat,
                                             focals_pixels=None,
                                             return_depth=return_depth,
                                             squre_clip=squre_clip,
                                            )
        else: # here 
            gaussian_splats = self.gaussian_predictor(x_input,
                                             view_to_world_transforms,
                                             source_cv2wT_quat,
                                             focals_pixels=None,
                                             return_depth=return_depth,
                                             squre_clip=squre_clip,
                                             unet_depth=unet_depth
                                            )
        
        gaussian_splat_batch = {k: v.contiguous() for k, v in gaussian_splats.items()}

        if render: # false
            bs = background.shape[0]
            x_novel = []
            depth_novel = []
            for _ in range(bs):
                output_dic = self.renderer(gaussian_splat_batch, _,
                        world_view_transforms[_:_+1].contiguous(),
                        full_proj_transforms[_:_+1].contiguous(),
                        camera_centers[_:_+1].contiguous(),
                        background[_:_+1].contiguous(),
                        config)

                x_novel += [output_dic["render"].reshape(-1,3,image_size,image_size)]
                depth_novel += [output_dic["rendered_depth"].reshape(-1,1,image_size, image_size)]
            x = torch.concat(x_novel, dim=0) # [bs, 3, h, w]
            depth = torch.concat(depth_novel, dim=0) # [bs, 1, h, w]

            # x: [b,3,h,w]
            # depth: [b,3,h,w]
        else: # here
            x, depth = None, None

        if return_3d_features: # here
            return x, depth, gaussian_splat_batch
        return x
    
