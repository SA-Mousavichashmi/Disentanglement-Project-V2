"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import copy
import glob
import os

import numpy as np
from PIL import Image
import torch
from utils.io import find_optimal_num_workers


# import datasets.base
# import utils

COLOUR_BLACK = 0
COLOUR_WHITE = 1
DATASETS = [
    'dsprites', 'shapes3d', 'cars3d', 'celeba'
]

def get_dataset(dataset_name):
    """Selection function to assign a respective dataset to a query string.
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASETS:
        raise NotImplementedError(f'Unknown datasets {dataset_name}!')

    if dataset_name == 'dsprites':
        import datasets.dsprites
        return datasets.dsprites.DSprites
    if dataset_name == 'shapes3d':
        import datasets.shapes3d
        return datasets.shapes3d.Shapes3D
    if dataset_name == 'cars3d':
        import datasets.cars3d
        return datasets.cars3d.Cars3D
    if dataset_name == 'celeba':
        import datasets.celeba
        return datasets.celeba.CelebA

def get_img_size(dataset, img_size=None):
    """Return the correct image size."""
    if img_size is not None:
        return img_size
    
    # Get the dataset class
    dataset_class = get_dataset(dataset)
    return dataset_class.img_size

def get_background(dataset, background_color=None):
    """Return the image background color."""
    if background_color is not None:
        return background_color
    
    # Get the dataset class
    dataset_class = get_dataset(dataset)
    return dataset_class.background_color

def get_dataloaders(
    dataset, shuffle=False, device=torch.device('cuda'), root='n/a',
    batch_size=256, num_workers=4):
    """Creates and returns PyTorch dataloaders for the specified dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset to load (e.g., "mnist", "dsprites").
    shuffle : bool, optional
        Whether to shuffle the dataset. Defaults to False.
    device : torch.device, optional
        Device to use for pin_memory. Defaults to 'cuda'.
    root : str, optional
        Path to the dataset root directory. Defaults to 'n/a', which uses
        the dataset's default root.
    batch_size : int, optional
        Number of samples per batch. Defaults to 256.
    num_workers : int, optional
        Number of subprocesses to use for data loading. Defaults to 4.
        If set to -1, attempts to find the optimal number.

    Returns
    -------
    torch.utils.data.DataLoader
        The DataLoader instance for the dataset.
    """
    pin_memory = True if device.type != 'cpu' else False

    if root == 'n/a':
        dataset = get_dataset(dataset)()
    else:
        dataset = get_dataset(dataset)(root=root)

    if num_workers == -1: # Getting optimal num of workers for dataloader
        num_workers = find_optimal_num_workers(
            dataset=dataset, batch_size=batch_size, pin_memory=True)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        sampler=None), {}


def preprocess(root, size=(64, 64), img_format='JPEG', center_crop=None):
    """Preprocess a folder of images.

    This function was taken from https://github.com/YannDubs/disentangling-vae.

    Parameters
    ----------
    root : string
        Root directory of all images.
    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.
    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.
    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, '*' + ext))

    for img_path in imgs:
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)