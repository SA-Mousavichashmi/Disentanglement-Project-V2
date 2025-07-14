# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess

import h5py
import numpy as np
import sklearn.preprocessing
import torchvision

import datasets
import datasets.base

class Shapes3D(datasets.base.DisentangledDataset):
    """Shapes3D Dataset from [1].

    3dshapes is a dataset of 3D shapes procedurally generated from 6 ground truth independent 
    latent factors. These factors are floor colour (10), wall colour (10), object colour (10), size (8), type (4) and azimuth (15). 
    All possible combinations of these latents are present exactly once, generating N = 480000 total images.

    Notes
    -----
    - Link : https://storage.googleapis.com/3d-shapes
    - hard coded metadata because issue with python 3 loading of python 2

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Hyunjik Kim, Andriy Mnih (2018). Disentangling by Factorizing.

    """
    urls = {
        "train":
        "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
    }
    files = {"train": "3dshapes.h5"}
    factor_names = ('floorCol', 'wallCol', 'objCol', 'objSize', 'objType', 'objAzimuth')
    factor_sizes = np.array([10, 10, 10, 8, 4, 15])
    img_size = (3, 64, 64)
    background_color = datasets.COLOUR_WHITE
    factor_values = {
        'floorCol': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        'wallCol': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        'objCol': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        'objSize': np.linspace(0.75, 1.25, 8),
        'objType': np.array([0., 1., 2., 3.]),
        'objAzimuth': np.linspace(-30., 30., 15)
    }

    def __init__(self,
                 selected_factors,
                 not_selected_factors_index_value,
                 root='data/shapes3d/', 
                 **kwargs):
        """Initialize the Shapes3D dataset.
        
        This method loads the dataset from numpy files, extracts the images and latent values,
        and processes them according to the selected factors of variation. The dataset contains
        3D shapes with controlled variations in floor color, wall color, object color, size, type, and azimuth.
        
        Parameters
        ----------
        selected_factors : list of str
            List of factor names to include in the dataset. Must be a subset of 
            ('floorCol', 'wallCol', 'objCol', 'objSize', 'objType', 'objAzimuth').
            These factors will be used as the latent variables.
            
        not_selected_factors_index_value : dict
            Dictionary mapping non-selected factor names to the index of their fixed value.
            For example, {'floorCol': 0} would fix the floor color factor to its first possible value.
            
        root : str, default='data/shapes3d/'
            Root directory where the dataset will be downloaded and stored.
            
        **kwargs : 
            Additional arguments passed to the parent class constructor.
            May include:
            - subset (float): Fraction of data to use (between 0 and 1).
            - transforms (list): Additional transforms to apply to the images.
        
        Notes
        -----
        The dataset will be downloaded automatically if it doesn't exist at the specified root.
        """
        
        super().__init__(root, 
                         selected_factors,
                         not_selected_factors_index_value, 
                         [torchvision.transforms.ToTensor()], 
                         **kwargs)

        self.imgs = np.load(self.train_data.replace('.h5', '_imgs.npy'))
        self.factor_values = np.load(self.train_data.replace('.h5', '_labs.npy'))

        self.selected_img_indices = self._get_selected_img_indices()
        self.selected_imgs = self.imgs[self.selected_img_indices]
        self._process_factor_values()

        if self.subset < 1:
            n_samples = int(len(self.imgs) * self.subset)
            subset = np.random.choice(len(self.imgs), n_samples, replace=False)
            self.imgs = self.imgs[subset]
            self.factor_values = self.factor_values[subset]

    @property
    def name(self):
        """Name of the dataset."""
        return 'shapes3d'

    @property
    def kwargs(self):
        """Keyword arguments for the dataset."""
        return {
            'selected_factors': self.selected_factors,
            'not_selected_factors_index_value': self.not_selected_factors_index_value,
            'root': self.root,
            'subset': self.subset
        }

    def download(self):
        """Download the dataset."""
        if not os.path.exists(os.path.join(self.train_data, '3dshapes.h5')):     
            os.makedirs(self.root)
            subprocess.check_call([
                "curl", "-L",
                type(self).urls["train"], "--output", self.train_data
            ]) 
        # For faster loading, a numpy copy will be created (reduces loading times by 300% at the cost of more storage).
        with h5py.File(self.train_data, 'r') as dataset:
            imgs = dataset['images'][()]
            factor_values = dataset['labels'][()]
        np.save(self.train_data.replace('.h5', '_imgs.npy'), imgs)
        np.save(self.train_data.replace('.h5', '_labs.npy'), factor_values)

    def __getitem__(self, idx):
        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        return self.transforms(self.selected_imgs[idx]), self.factor_values[idx]
