import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from packaging import version as pver
from torchvision import transforms as T, utils

from PIL import Image, ImageOps
from typing import List, Tuple

from src.camera import compute_cam2world_matrix, normalize, sample_rays
from src.utils import TensorGroup
import random

class Imagenet_Dataset_demo:
    def __init__(self, folder, image_size=64, config=None, random_flip=True, get_224=False):
        self.random_flip = random_flip
        if not config['dataset_params'].get('all_classes', False):
            self.image_names = glob.glob(folder + "*.jpg")#[:16]
        else:
            # get the paths for the image files
            # can be further improved
            if folder.endswith('txt'):
                self.image_names = [i.strip() for i in open(folder, 'r').readlines()]
                print(self.image_names )
            else:
                self.image_names = []
                folder_names = os.listdir(folder)
                for f in folder_names:
                    self.image_names.extend(glob.glob(os.path.join(folder, f) + "/*.jpg"))
        # print('Folder: ', not config['dataset_params'].get('all_classes', False))
        # print('Folder: ', folder)
        # print('Folder: ', folder_names)
        print('Number of files: ', len(self.image_names))
        self.rotate = T.functional.rotate
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=Image.LANCZOS),
            np.array,
            T.ToTensor()
        ])
        self.get_224 = get_224
        if get_224:
            output_size = 224
            print('**Getting Size: ', output_size)
            self.transform_224 = T.Compose([
                T.Resize(output_size, interpolation=Image.LANCZOS),
                np.array,
                T.ToTensor()
            ])

        self.fov = 18
        self.radius = 5
        fov_degrees = torch.ones(1,1)*self.fov
        self.camera_params = TensorGroup(
                angles=torch.zeros(1,3),
                # fov=0,
                radius=torch.ones(1,1)*self.radius,
                look_at=torch.zeros(1,3),
            )

        self.camera_params.angles[:, 0] = np.pi/2#+np.pi/18
        self.camera_params.angles[:, 1] = np.pi/2#+np.pi/18

        cam2w = compute_cam2world_matrix(self.camera_params)
        ray_origins, ray_directions = sample_rays(cam2w, fov_degrees, [image_size,image_size])
        self.rays_o = ray_origins.reshape(-1,3)
        self.rays_d = ray_directions.reshape(-1,3)
        self.depth_type = 'leres' # 

        self.config = config  

    def __len__(self):
        return len(self.image_names)

    def get_names():
        return self.image_names

    def __getitem__(self, idx):
        img_filename = self.image_names[idx]
        if self.depth_type == 'leres':
            depth_filename = img_filename.replace('.jpg', '_depth.png')

        results = {}
        results['idx'] = idx
        results['name'] = img_filename.split('/')[-1]
        images = Image.open(img_filename)
        # results['filename'] = img_filename
        results['depth'] = Image.open(depth_filename)

        results['images'] = self.transform(images)
        results['depth'] = self.transform(results['depth'])
        results['depth'] = results['depth'] / 65536 * 2.0 + 4
        if self.get_224:
            results['images_224'] = self.transform_224(images)

        if self.random_flip:
            if random.random() < 0.5:
                results['idx'] = idx + len(self.image_names)
                results['images'] = T.functional.hflip(results['images'])
                results['depth'] = T.functional.hflip(results['depth'])
                if self.get_224:
                    results['images_224'] = T.functional.hflip(results['images_224'])

        return results

