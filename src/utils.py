# Code adopted from various libraries which include Eg3D, 3DGP(please use citations from main paper)

from prettytable import PrettyTable
import ctypes
import fnmatch
import importlib
import inspect
import numpy as np
import os
import shutil
import sys
import types
import io
import pickle
import re
import requests
import html
import hashlib
import glob
import tempfile
import urllib
import urllib.request
import uuid

import torch
from distutils.util import strtobool
from typing import Any, List, Tuple, Union, Callable, Dict
from omegaconf import DictConfig
import matplotlib
# helpers functions


# def sample_front_circle(camera_params, num_frames, fov_diff=0.0, yaw_diff=0.25, pitch_diff=0.1):
def sample_front_circle(camera_params, num_frames, fov_diff=0.0, yaw_diff=0.25, pitch_diff=0.15):
    trajectory = EasyDict({'name': 'front_circle', 'num_frames': num_frames, 'fov_diff': fov_diff, 'yaw_diff': yaw_diff, 'pitch_diff': pitch_diff, 'use_mean_camera': True})

    num_samples = len(camera_params)
    num_frames = trajectory.num_frames
    camera_params = camera_params.repeat_interleave(num_frames, dim=0) # [num_samples * num_frames, ...]

    if trajectory.name == 'front_circle':
        steps = torch.linspace(0, 1, num_frames).repeat_interleave(num_samples) # [num_samples * num_frames]
        yaw = camera_params.angles[:, 0].cpu() + trajectory.yaw_diff * torch.sin(steps * 2 * np.pi) # [num_samples * num_frames]
        pitch = camera_params.angles[:, 1].cpu() + trajectory.pitch_diff * torch.cos(steps * 2 * np.pi) # [num_samples * num_frames]
        angles = torch.stack([yaw, pitch, camera_params.angles[:, 2].cpu()], dim=1) # [num_samples * num_frames, 3]
        fov = (camera_params.fov.cpu() + trajectory.fov_diff * torch.sin(steps * 2 * np.pi)) # [num_samples * num_frames]
        # print('fov: ', fov)
    elif trajectory.name == 'line':
        yaws = torch.linspace(trajectory.yaw_start, trajectory.yaw_end, num_frames).repeat(num_samples) # [num_samples * num_frames]
        pitches = torch.linspace(trajectory.pitch_start, trajectory.pitch_end, num_frames).repeat(num_samples) # [num_samples * num_frames]
        angles = torch.stack([yaws, pitches, torch.zeros_like(yaws)], axis=1) # [num_samples * num_frames, 3]
        fov = camera_params.fov.cpu() if trajectory.fov is None else (torch.ones_like(camera_params.fov.cpu()) * trajectory.fov)  # [num_samples * num_frames]

    camera_samples = TensorGroup(
        angles=angles,
        fov=fov + trajectory.get('fov_offset', 0.0),
        radius=camera_params.radius.cpu(),
        look_at=camera_params.look_at.cpu(),
    ) # [num_samples * num_frames, ...]
    # camera_samples = camera_samples.reshape_each(lambda x: [num_samples, num_frames, *x.shape[1:]])
    return camera_samples


def sample_front_circle_gs(camera_params, num_frames, fov_diff=0.0, yaw_diff=0.25, pitch_diff=0.1):
    trajectory = EasyDict({'name': 'front_circle', 'num_frames': num_frames, 'fov_diff': fov_diff, 'yaw_diff': yaw_diff, 'pitch_diff': pitch_diff, 'use_mean_camera': True})

    num_samples = len(camera_params)
    num_frames = trajectory.num_frames
    camera_params = camera_params.repeat_interleave(num_frames, dim=0) # [num_samples * num_frames, ...]

    if trajectory.name == 'front_circle':
        steps = torch.linspace(0, 1, num_frames).repeat_interleave(num_samples) # [num_samples * num_frames]
        yaw = camera_params.angles[:, 0].cpu() - trajectory.yaw_diff * torch.sin(steps * 2 * np.pi) # [num_samples * num_frames]
        pitch = camera_params.angles[:, 1].cpu() + trajectory.pitch_diff * torch.cos(steps * 2 * np.pi) # [num_samples * num_frames]
        angles = torch.stack([yaw, pitch, camera_params.angles[:, 2].cpu()], dim=1) # [num_samples * num_frames, 3]
        fov = (camera_params.fov.cpu() + trajectory.fov_diff * torch.sin(steps * 2 * np.pi)) # [num_samples * num_frames]
    elif trajectory.name == 'line':
        yaws = torch.linspace(trajectory.yaw_start, trajectory.yaw_end, num_frames).repeat(num_samples) # [num_samples * num_frames]
        pitches = torch.linspace(trajectory.pitch_start, trajectory.pitch_end, num_frames).repeat(num_samples) # [num_samples * num_frames]
        angles = torch.stack([yaws, pitches, torch.zeros_like(yaws)], axis=1) # [num_samples * num_frames, 3]
        fov = camera_params.fov.cpu() if trajectory.fov is None else (torch.ones_like(camera_params.fov.cpu()) * trajectory.fov)  # [num_samples * num_frames]

    camera_samples = TensorGroup(
        angles=angles,
        fov=fov + trajectory.get('fov_offset', 0.0),
        radius=camera_params.radius.cpu(),
        look_at=camera_params.look_at.cpu(),
    ) # [num_samples * num_frames, ...]
    # camera_samples = camera_samples.reshape_each(lambda x: [num_samples, num_frames, *x.shape[1:]])
    return camera_samples



def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img



def colorize_first(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img, vmin, vmax




def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    @staticmethod
    def init_recursively(value: Union[Dict, DictConfig]):
        if not isinstance(value, (Dict, DictConfig, EasyDict)):
            return value
        else:
            return EasyDict(**{k: EasyDict.init_recursively(v) for k, v in value.items()})

#----------------------------------------------------------------------------

class TensorGroup(EasyDict):
    """
    Sometimes, it is very convenient to group tensors into a group.
    You can slice/split TensorDict in the same manner as you would do with normal torch tensors
    The tensors are aligned via the first axis.

    Caution: when updating the properties, it's your responsibility to
    make sure that the shapes remain to be correct (i.e. of the same length).
    """
    def __init__(self, **kwargs):
        keys = list(kwargs.keys())
        values = list(kwargs.values())

        assert len(keys) == len(values)
        assert all(isinstance(key, str) for key in keys), f"Wrong types for keys: {keys}"
        assert all((isinstance(t, torch.Tensor) or isinstance(t, TensorGroup)) for t in values), f"Wrong types for values: {dict(zip(keys, [type(v) for v in values]))}"
        assert all(len(t) == len(values[0]) for t in values), f"Wrong shapes: {dict(zip(keys, [v.shape for v in values]))}"
        assert all(t.device == values[0].device for t in values), f"All tensor should be on the same device, but got: {dict(zip(keys, [v.device for v in values]))}"
        assert all(not field in keys for field in ['_length', 'shape'])

        self._length = values[0].shape[0]
        self.shape = [len(self), None]

        super(TensorGroup, self).__init__(**kwargs)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, item: Any):
        if isinstance(item, str):
            return super(TensorGroup, self).__getitem__(item)
        else:
            return TensorGroup(**{k: v[item] for k, v in self.items()})

    def items(self) -> List[Tuple[str, Union[torch.Tensor, "TensorGroup"]]]:
        return [(k, v) for k, v in super(TensorGroup, self).items() if not k in ('_length', 'shape')]

    def keys(self) -> List[str]:
        return [k for k, _ in self.items()]

    def values(self) -> List[Union[torch.Tensor, "TensorGroup"]]:
        return [v for _, v in self.items()]

    def split(self, group_size: int) -> List["TensorGroup"]:
        result = []
        for group_idx in range((len(self) + group_size - 1) // group_size):
            result.append(self[group_idx * group_size: (group_idx + 1) * group_size])
        return result

    def max(self) -> torch.Tensor:
        """
        This method is useless on its own and we use it simply to aggregate some values
        from all the tensors to have DDP consistency
        """
        return torch.stack([v.max() for k, v in self.items()]).max() # [1]

    def reduce_mean(self) -> torch.Tensor:
        """
        This method is useless on its own and we use it simply to aggregate some values
        from all the tensors to have DDP consistency
        """
        return self.sum() / self.numel() # [1]

    def sum(self) -> torch.Tensor:
        return torch.stack([v.sum() for k, v in self.items()]).sum() # [1]

    def numel(self) -> int:
        return sum([v.numel() for k, v in self.items()])

    def to(self, *args, **kwargs) -> "TensorGroup":
        return TensorGroup(**{k: v.to(*args, **kwargs) for k, v in self.items()})

    def clone(self) -> "TensorGroup":
        return TensorGroup(**{k: v.clone() for k, v in self.items()})

    def repeat(self, num, dim) -> "TensorGroup":
        def repeat_dims(t, num, dim):
            dims = [1,]*len(t.shape)
            dims[dim] = num
            return dims
        return TensorGroup(**{k: v.repeat(repeat_dims(v, num, dim)) for k, v in self.items()})

    def repeat_interleave(self, *args, **kwargs) -> "TensorGroup":
        return TensorGroup(**{k: v.repeat_interleave(*args, **kwargs) for k, v in self.items()})

    def __add__(self, other: Any) -> "TensorGroup":
        if isinstance(other, TensorGroup):
            return TensorGroup(**{k: (v + other[k]) for k, v in self.items()})
        else:
            return TensorGroup(**{k: (v + other) for k, v in self.items()})

    def __radd__(self, other) -> "TensorGroup":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "TensorGroup":
        if isinstance(other, TensorGroup):
            return TensorGroup(**{k: (v - other[k]) for k, v in self.items()})
        else:
            return TensorGroup(**{k: (v - other) for k, v in self.items()})

    def __pow__(self, other: Any) -> "TensorGroup":
        if isinstance(other, TensorGroup):
            return TensorGroup(**{k: (v ** other[k]) for k, v in self.items()})
        else:
            return TensorGroup(**{k: (v ** other) for k, v in self.items()})

    def __mul__(self, other: Any) -> "TensorGroup":
        if isinstance(other, TensorGroup):
            return TensorGroup(**{k: (v * other[k]) for k, v in self.items()})
        else:
            return TensorGroup(**{k: (v * other) for k, v in self.items()})

    def __rmul__(self, other: Any) -> "TensorGroup":
        return self.__mul__(other)

    def float(self) -> "TensorGroup":
        return TensorGroup(**{k: v.float() for k, v in self.items()})

    @property
    def device(self):
        return next(iter(self.items()))[1].device

    @property
    def shapes(self) -> List[torch.Size]:
        return [v.shape for v in self.values()]

    def detach(self) -> "TensorGroup":
        return TensorGroup(**{k: v.detach() for k, v in self.items()})

    def reshape_each(self, reshaper: Callable) -> "TensorGroup":
        return TensorGroup(**{k: v.reshape(reshaper(v)) for k, v in self.items()})

    def cpu(self) -> "TensorGroup":
        return TensorGroup(**{k: v.cpu() for k, v in self.items()})

    def clamp(self, *args, **kwargs) -> "TensorGroup":
        return TensorGroup(**{k: v.clamp(*args, **kwargs) for k, v in self.items()})

    def permute(self, *args, **kwargs) -> "TensorGroup":
        return TensorGroup(**{k: v.permute(*args, **kwargs) for k, v in self.items()})

    def mean(self, *args, **kwargs) -> "TensorGroup":
        return TensorGroup(**{k: v.mean(*args, **kwargs) for k, v in self.items()})

    @staticmethod
    def cat(tgroups: List["TensorGroup"], dim: int=0):
        keys_set = set(tgroups[0].keys())
        assert [set(tg.keys()) == keys_set for tg in tgroups], f"Keys should be the same: {[list(tg.keys()) for tg in tgroups]}"
        return TensorGroup(**{k: torch.cat([tg[k] for tg in tgroups], dim=dim) for k in keys_set})

#----------------------------------------------------------------------------

class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None


# Cache directories
# ------------------------------------------------------------------------------------------

_dnnlib_cache_dir = None

def set_cache_dir(path: str) -> None:
    global _dnnlib_cache_dir
    _dnnlib_cache_dir = path

def make_cache_dir_path(*paths: str) -> str:
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)

# Small util functions
# ------------------------------------------------------------------------------------------


def format_time(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)


def format_time_brief(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m".format(s // (60 * 60), (s // 60) % 60)
    else:
        return "{0}d {1:02}h".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24)


def ask_yes_no(question: str) -> bool:
    """Ask the user the question until the user inputs a valid answer."""
    while True:
        try:
            print("{0} [y/n]".format(question))
            return strtobool(input().lower())
        except ValueError:
            pass


def tuple_product(t: Tuple) -> Any:
    """Calculate the product of the tuple elements."""
    result = 1

    for v in t:
        result *= v

    return result


_str_to_ctype = {
    "uint8": ctypes.c_ubyte,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "int8": ctypes.c_byte,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double
}


def get_dtype_and_ctype(type_obj: Any) -> Tuple[np.dtype, Any]:
    """Given a type name string (or an object having a __name__ attribute), return matching Numpy and ctypes types that have the same size in bytes."""
    type_str = None

    if isinstance(type_obj, str):
        type_str = type_obj
    elif hasattr(type_obj, "__name__"):
        type_str = type_obj.__name__
    elif hasattr(type_obj, "name"):
        type_str = type_obj.name
    else:
        raise RuntimeError("Cannot infer type name from input")

    assert type_str in _str_to_ctype.keys()

    my_dtype = np.dtype(type_str)
    my_ctype = _str_to_ctype[type_str]

    assert my_dtype.itemsize == ctypes.sizeof(my_ctype)

    return my_dtype, my_ctype


def is_pickleable(obj: Any) -> bool:
    try:
        with io.BytesIO() as stream:
            pickle.dump(obj, stream)
        return True
    except:
        return False


# Functionality to import modules/objects by name, and call functions by name
# ------------------------------------------------------------------------------------------

def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed)."""

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name) # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == '':
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: str = None, **kwargs) -> Any:
    """Finds the python class with the given name and constructs it with the given arguments."""
    return call_func_by_name(*args, func_name=class_name, **kwargs)


def get_module_dir_by_obj_name(obj_name: str) -> str:
    """Get the directory path of the module containing the given object name."""
    module, _ = get_module_from_obj_name(obj_name)
    return os.path.dirname(inspect.getfile(module))


def is_top_level_function(obj: Any) -> bool:
    """Determine whether the given object is a top-level function, i.e., defined at module scope using 'def'."""
    return callable(obj) and obj.__name__ in sys.modules[obj.__module__].__dict__


def get_top_level_function_name(obj: Any) -> str:
    """Return the fully-qualified name of a top-level function."""
    assert is_top_level_function(obj)
    module = obj.__module__
    if module == '__main__':
        module = os.path.splitext(os.path.basename(sys.modules[module].__file__))[0]
    return module + "." + obj.__name__


# File system helpers
# ------------------------------------------------------------------------------------------

def list_dir_recursively_with_ignore(dir_path: str, ignores: List[str] = None, add_base_to_relative: bool = False) -> List[Tuple[str, str]]:
    """List all files recursively in a given directory while ignoring given file and directory names.
    Returns list of tuples containing both absolute and relative paths."""
    assert os.path.isdir(dir_path)
    base_name = os.path.basename(os.path.normpath(dir_path))

    if ignores is None:
        ignores = []

    result = []

    for root, dirs, files in os.walk(dir_path, topdown=True):
        for ignore_ in ignores:
            dirs_to_remove = [d for d in dirs if fnmatch.fnmatch(d, ignore_)]

            # dirs need to be edited in-place
            for d in dirs_to_remove:
                dirs.remove(d)

            files = [f for f in files if not fnmatch.fnmatch(f, ignore_)]

        absolute_paths = [os.path.join(root, f) for f in files]
        relative_paths = [os.path.relpath(p, dir_path) for p in absolute_paths]

        if add_base_to_relative:
            relative_paths = [os.path.join(base_name, p) for p in relative_paths]

        assert len(absolute_paths) == len(relative_paths)
        result += zip(absolute_paths, relative_paths)

    return result


def copy_files_and_create_dirs(files: List[Tuple[str, str]]) -> None:
    """Takes in a list of tuples of (src, dst) paths and copies files.
    Will create all necessary directories."""
    for file in files:
        target_dir_name = os.path.dirname(file[1])

        # will create all intermediate-level directories
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)

        shutil.copyfile(file[0], file[1])


# URL helpers
# ------------------------------------------------------------------------------------------

def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)
