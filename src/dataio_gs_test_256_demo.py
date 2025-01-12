import glob
import os

import numpy as np
import torch
from torchvision import transforms as T, utils

from PIL import Image, ImageOps
# from typing import List, Tuple

from src.camera import compute_cam2world_matrix
from src.utils import TensorGroup
import random
import math
from torchvision.transforms import functional as F

# class PILToTensor:
    # def __call__(self, image):
        # image = F.pil_to_tensor(image)
        # image = torch.from_numpy(np.array(image))
        # return image

class ImagenetGS_Dataset_test_256_demo:
    def __init__(self, folder, image_size=64, config=None, random_flip=True, get_224=False):
        self.random_flip = random_flip
        # print('self.random_flip', self.random_flip)
        self.norm_depth = config['dataset_params'].get('norm_depth', False)
        # print('self.norm_depth: ', self.norm_depth)
        self.norm_depth_to01 = config['dataset_params'].get('norm_depth_to01', False)
        # print('self.norm_depth_to01: ', self.norm_depth_to01)

        self.image_size = image_size
        
        if not config['dataset_params'].get('all_classes', False): 
            if folder.endswith('txt'):
                self.image_names = [i.strip() for i in open(folder, 'r').readlines()]
                print(self.image_names )
            else:
                self.image_names = glob.glob(folder + "*.jpg")
        else: # here
            # # get the paths for the image files
            # # can be further improved

            # fixed files for demo
            if folder.endswith('txt'):
                self.image_names = [i.strip() for i in open(folder, 'r').readlines()]
                print(self.image_names )
            else:
                # for images_cat
                print('load from folder: ', folder)
                self.image_names = glob.glob(folder + "/*")
                self.image_names = sorted([i for i in self.image_names if not i.endswith('_depth.png')])
                print(self.image_names)

        print('Number of files: ', len(self.image_names))
        self.rotate = T.functional.rotate
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=Image.LANCZOS),
            np.array,
            T.ToTensor()
        ])
        self.transform_normal = T.Compose([
            T.ToPILImage(),
            T.Resize(image_size, interpolation=Image.BILINEAR),
            T.ToTensor()
        ])
        self.transform_resize = T.Resize(image_size, interpolation=Image.LANCZOS)
        self.get_224 = get_224
        if get_224:
            output_size = 224
            print('**Getting Size: ', output_size)
            self.transform_224 = T.Compose([
                T.Resize(output_size, interpolation=Image.LANCZOS),
                np.array,
                T.ToTensor()
            ])

        self.fov = config["model"]["fov"] # 18 by default
        self.radius = config["model"]["radius"]
        self.look_at = config["model"]["look_at"]
        fov_degrees = torch.ones(1,1)*self.fov
        self.camera_params = TensorGroup(
                angles=torch.zeros(1,3),
                # fov=0,
                radius=torch.ones(1,1)*self.radius,
                look_at=torch.zeros(1,3),
            )

        self.camera_params.look_at[:,2] = self.look_at

        cam2w = compute_cam2world_matrix(self.camera_params) # [bs=1, 4, 4]  
        cam2w = torch.inverse(cam2w)

        # print('cam2w: ', cam2w)
        self.cam2w = cam2w
        

        Rt = torch.inverse(cam2w) # [bs=1, 4, 4]
        self.world_view_transforms = Rt.transpose(1,2) # [bs=1, 4, 4]
        self.view_to_world_transforms = cam2w.transpose(1,2) # [bs=1, 4, 4]
        self.camera_centers = self.world_view_transforms.inverse()[:, 3, :3] # [bs=1, 3]


        self.projection_matrix = getProjectionMatrix(
            znear=config["dataset_params"]["z_near"], zfar=config["dataset_params"]["z_far"],
            fovX=self.fov * 2 * np.pi / 360, # pi/3
            fovY=self.fov * 2 * np.pi / 360).transpose(0,1)
        self.full_proj_transforms = (self.world_view_transforms.bmm(self.projection_matrix.unsqueeze(0))) # [bs=1, 4, 4]        

        self.world_view_transforms_ori = self.world_view_transforms.clone()
        self.view_to_world_transforms_ori = self.view_to_world_transforms.clone()
        self.full_proj_transforms_ori = self.full_proj_transforms.clone()
        self.camera_centers_ori = self.camera_centers.clone()


        # # update camera pose
        self.inverse_first_camera = None
        if config['opt']['update_pose']:
            self.world_view_transforms, \
            self.view_to_world_transforms, \
            self.full_proj_transforms, \
            self.camera_centers, \
            self.inverse_first_camera \
                = update_camera_pose(
                self.world_view_transforms, \
                self.view_to_world_transforms, \
                self.full_proj_transforms, \
                self.camera_centers,
                None,
                first=True
            )

        self.source_cv2wT_quat = matrix_to_quaternion(self.view_to_world_transforms[0,:3,:3].transpose(0,1)).unsqueeze(0) # [bs=1, 1, 4] 

        # print('self.world_view_transforms: ', self.world_view_transforms)
        # print('self.view_to_world_transforms: ', self.view_to_world_transforms)
        # print('self.full_proj_transforms: ', self.full_proj_transforms)
        # print('self.camera_centers: ', self.camera_centers)
        # print('self.inverse_first_camera: ', self.inverse_first_camera)
        # print('self.source_cv2wT_quat: ', self.source_cv2wT_quat)

        self.depth_type = config['opt']['depth_type'] # 
        self.normal_type = config['opt']['normal_type'] # 

        self.config = config  

    def __len__(self):
        return len(self.image_names)

    def get_names(self):
        return self.image_names

    def __getitem__(self, idx):
        img_filename = self.image_names[idx]
        results = {}
        results['idx'] = idx
        results['name'] = img_filename.split('/')[-1]
        images = Image.open(img_filename).convert('RGB') # For cat iamges
        results['images'] = self.transform(images)

        if self.depth_type == 'leres':
            if img_filename.endswith('.jpg'):
                depth_filename = img_filename.replace('.jpg', '_depth.png')
            elif img_filename.endswith('.png'):
                depth_filename = img_filename.replace('.png', '_depth.png')

            results['depth'] = Image.open(depth_filename).convert('I')
            results['depth'] = self.transform(results['depth'])
            # results['depth'] = results['depth'] / 65536 * 2.0 + self.config['dataset_params']['z_near'] # 0-4
            results['depth'] = results['depth'] / 65536 
            if self.norm_depth_to01: # 0-1
                results['depth'] = (results['depth'] - results['depth'].min()) / (results['depth'].max() - results['depth'].min())
            results['depth'] = results['depth'] * 2.0 + self.config['dataset_params']['z_near'] # 0-4
        elif self.depth_type == 'marigold':
            depth_filename = os.path.join(os.path.dirname(img_filename.replace('imagenet_256_with_depth', 'imagenet_256_marigold')), 'depth_npy', results['name'].replace('.jpg', '_pred.npy'))
            results['depth'] = Image.fromarray(np.load(depth_filename)) # 0 - 1
            results['depth'] = self.transform(results['depth'])
            if self.norm_depth_to01: # 0-1
                results['depth'] = (results['depth'] - results['depth'].min()) / (results['depth'].max() - results['depth'].min())
            results['depth'] = results['depth'] * 2.0 + self.config['dataset_params']['z_near'] # 0-4
        elif self.depth_type == 'da2':
            depth_filename = os.path.join(os.path.dirname(img_filename.replace('imagenet_256_with_depth', 'imagenet_256_depth-anything-v2')), 'pred_npy', results['name'].replace('.jpg', '_depth.npy'))
            # v5: 
            disp = np.load(depth_filename) / 255.0
            depth = 1 / np.maximum(disp, 0.1) # 
            depth = np.power (depth, 1.0 / 50)
            results['depth'] = Image.fromarray(depth) 

            results['depth'] = self.transform(results['depth']) 
            if self.norm_depth_to01: # 0-1 
                results['depth'] = (results['depth'] - results['depth'].min()) / (results['depth'].max() - results['depth'].min()) 
            # results['depth'] = results['depth'] / 255.0 * 2.0 + self.config['dataset_params']['z_near'] # 0-4 
            results['depth'] = results['depth'] * 2.0 + self.config['dataset_params']['z_near'] # 0-4 

        # # load normal:
        # if self.normal_type == 'omni':
        #     normal_filename = os.path.join(os.path.dirname(img_filename.replace('imagenet_256_with_depth', 'imagenet_256_omni')), 'pred_npy', results['name'].replace('.jpg', '.npy'))
        #     normal = np.load(normal_filename) # (3, 256, 256) float32 0.013078504 1.0
        #     normal = normal # chw, 0,1
        #     # results['normal'] = Image.fromarray(normal) 
        #     results['normal'] = torch.from_numpy(normal)
        #     # print('np: ', normal.shape, normal.dtype, normal.min(), normal.max())
        #     results['normal'] = self.transform_normal(results['normal']) * 2.0 - 1

        # elif self.normal_type == 'dsine':
        #     normal_filename = os.path.join(os.path.dirname(img_filename.replace('imagenet_256_with_depth', 'imagenet_256_dsine')), 'pred_npy', results['name'].replace('.jpg', '.npy'))
        #     normal = np.load(normal_filename) # (256, 256, 3) float32 0.79565454 254.99979
        #     normal = ((255.0 - normal.transpose([2,0,1])) / 255.0) # chw, 0,1
        #     # results['normal'] = Image.fromarray(normal) 
        #     results['normal'] = torch.from_numpy(normal)
        #     # print('np: ', normal.shape, normal.dtype, normal.min(), normal.max())
        #     results['normal'] = self.transform_normal(results['normal']) * 2.0 - 1
        
        # # print('normal: ', results['normal'].shape, results['normal'].dtype, results['normal'].min(), results['normal'].max())


        if self.norm_depth: # 0-1 
            results['depth'] = (results['depth'] - results['depth'].min()) / (results['depth'].max() - results['depth'].min())
        if self.get_224: 
            results['images_224'] = self.transform_224(images) 

        if self.random_flip:
            if random.random() < 0.5:
                results['idx'] = idx + len(self.image_names)
                results['images'] = T.functional.hflip(results['images'])
                results['depth'] = T.functional.hflip(results['depth'])
                # results['normal'] = T.functional.hflip(results['normal'])
                if self.get_224:
                    results['images_224'] = T.functional.hflip(results['images_224'])
        
        # print('depth: ', results['depth'].shape, results['depth'].dtype, results['depth'].min(), results['depth'].max())
        # print('images: ', results['images'].shape, results['images'].dtype, results['images'].min(), results['images'].max())

        return results 


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    # P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 2] = z_sign * (znear + zfar) / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    # P[2, 2] = z_sign * (znear + zfar) / (zfar - znear)
    # P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
    return P

def matrix_to_quaternion(M: torch.Tensor) -> torch.Tensor:
    """
    Matrix-to-quaternion conversion method. Equation taken from 
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    Args:
        M: rotation matrices, (3 x 3)
    Returns:
        q: quaternion of shape (4)
    """
    tr = 1 + M[ 0, 0] + M[ 1, 1] + M[ 2, 2]

    if tr > 0:
        r = torch.sqrt(tr) / 2.0
        x = ( M[ 2, 1] - M[ 1, 2] ) / ( 4 * r )
        y = ( M[ 0, 2] - M[ 2, 0] ) / ( 4 * r )
        z = ( M[ 1, 0] - M[ 0, 1] ) / ( 4 * r )
    elif ( M[ 0, 0] > M[ 1, 1]) and (M[ 0, 0] > M[ 2, 2]):
        S = torch.sqrt(1.0 + M[ 0, 0] - M[ 1, 1] - M[ 2, 2]) * 2 # S=4*qx 
        r = (M[ 2, 1] - M[ 1, 2]) / S
        x = 0.25 * S
        y = (M[ 0, 1] + M[ 1, 0]) / S 
        z = (M[ 0, 2] + M[ 2, 0]) / S 
    elif M[ 1, 1] > M[ 2, 2]: 
        S = torch.sqrt(1.0 + M[ 1, 1] - M[ 0, 0] - M[ 2, 2]) * 2 # S=4*qy
        r = (M[ 0, 2] - M[ 2, 0]) / S
        x = (M[ 0, 1] + M[ 1, 0]) / S
        y = 0.25 * S
        z = (M[ 1, 2] + M[ 2, 1]) / S
    else:
        S = torch.sqrt(1.0 + M[ 2, 2] - M[ 0, 0] -  M[ 1, 1]) * 2 # S=4*qz
        r = (M[ 1, 0] - M[ 0, 1]) / S
        x = (M[ 0, 2] + M[ 2, 0]) / S
        y = (M[ 1, 2] + M[ 2, 1]) / S
        z = 0.25 * S

    return torch.stack([r, x, y, z], dim=-1)


def update_camera_pose(world_view_transforms, view_to_world_transforms, full_proj_transforms, camera_centers, inverse_first_camera=None, first=False):

    if first and world_view_transforms.shape[0] == 1:
        # print('update_camera_pose, init')
        assert inverse_first_camera is None
        inverse_first_camera = world_view_transforms[0].inverse().clone()

        new_world_view_transforms = torch.zeros_like(world_view_transforms)
        new_view_to_world_transforms = torch.zeros_like(view_to_world_transforms)
        new_full_proj_transforms = torch.zeros_like(full_proj_transforms)
        new_camera_centers = torch.zeros_like(camera_centers)

        for c in range(world_view_transforms.shape[0]):
            if world_view_transforms.shape[0] != 1:
                print('debug: ', inverse_first_camera.shape, world_view_transforms.shape)
            new_world_view_transforms[c] = torch.bmm(
                                                inverse_first_camera.unsqueeze(0), # 1 4 4
                                                world_view_transforms[c].unsqueeze(0)).squeeze(0)
            new_view_to_world_transforms[c] = torch.bmm(
                                                view_to_world_transforms[c].unsqueeze(0),
                                                inverse_first_camera.inverse().unsqueeze(0)).squeeze(0)
            new_full_proj_transforms[c] = torch.bmm(
                                                inverse_first_camera.unsqueeze(0),
                                                full_proj_transforms[c].unsqueeze(0)).squeeze(0)
            # new_camera_centers[c] = world_view_transforms[c].inverse()[3, :3]
            new_camera_centers[c] = new_world_view_transforms[c].inverse()[3, :3]


        return new_world_view_transforms, new_view_to_world_transforms, new_full_proj_transforms, new_camera_centers, inverse_first_camera        

    else:
        assert inverse_first_camera is not None

        if len(inverse_first_camera.shape) == 2: # [4,4]
            new_world_view_transforms = torch.zeros_like(world_view_transforms)
            new_view_to_world_transforms = torch.zeros_like(view_to_world_transforms)
            new_full_proj_transforms = torch.zeros_like(full_proj_transforms)
            new_camera_centers = torch.zeros_like(camera_centers)

            for c in range(world_view_transforms.shape[0]):
                new_world_view_transforms[c,0] = torch.bmm(
                                                    inverse_first_camera.unsqueeze(0), # 1 4 4
                                                    world_view_transforms[c,0].unsqueeze(0)).squeeze(0)
                new_view_to_world_transforms[c,0] = torch.bmm(
                                                    view_to_world_transforms[c,0].unsqueeze(0),
                                                    inverse_first_camera.inverse().unsqueeze(0)).squeeze(0)
                new_full_proj_transforms[c,0] = torch.bmm(
                                                    inverse_first_camera.unsqueeze(0),
                                                    full_proj_transforms[c,0].unsqueeze(0)).squeeze(0)
                new_camera_centers[c] = new_world_view_transforms[c,0].inverse()[3, :3]


            return new_world_view_transforms, new_view_to_world_transforms, new_full_proj_transforms, new_camera_centers, inverse_first_camera        

        elif len(inverse_first_camera.shape) == 3:
            new_world_view_transforms = torch.zeros_like(world_view_transforms)
            new_view_to_world_transforms = torch.zeros_like(view_to_world_transforms)
            new_full_proj_transforms = torch.zeros_like(full_proj_transforms)
            new_camera_centers = torch.zeros_like(camera_centers)

            for c in range(world_view_transforms.shape[0]):

                new_world_view_transforms[c,0] = torch.bmm(
                                                    inverse_first_camera[c].unsqueeze(0), # 1 4 4
                                                    world_view_transforms[c,0].unsqueeze(0)).squeeze(0)
                new_view_to_world_transforms[c,0] = torch.bmm(
                                                    view_to_world_transforms[c,0].unsqueeze(0),
                                                    inverse_first_camera[c].inverse().unsqueeze(0)).squeeze(0)
                new_full_proj_transforms[c,0] = torch.bmm(
                                                    inverse_first_camera[c].unsqueeze(0),
                                                    full_proj_transforms[c,0].unsqueeze(0)).squeeze(0)
                new_camera_centers[c] = new_world_view_transforms[c,0].inverse()[3, :3]


            return new_world_view_transforms, new_view_to_world_transforms, new_full_proj_transforms, new_camera_centers, inverse_first_camera        





    

