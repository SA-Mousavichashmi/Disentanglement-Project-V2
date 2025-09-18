# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess

import numpy as np
import sklearn.preprocessing
import torchvision

import datasets
import datasets.base

class DSprites(datasets.base.DisentangledDataset):
    """DSprites Dataset from [1].

    Disentanglement test Sprites dataset.Procedurally generated 2D shapes, from 6
    disentangled latent factors. This dataset uses 6 latents, controlling the color,
    shape, scale, rotation and position of a sprite. All possible variations of
    the latents are present. Ordering along dimension 1 is fixed and can be mapped
    back to the exact latent values that generated that image. Pixel outputs are
    different. No noise added.

    Notes
    -----
    - Link : https://github.com/deepmind/dsprites-dataset/
    - hard coded metadata because issue with python 3 loading of python 2

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick,
        M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts
        with a constrained variational framework. In International Conference
        on Learning Representations.

    """
    urls = {
        "train":
        "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"
    }
    files = {"train": "dsprite_train.npz"}
    factor_names = ('color', 'shape', 'scale', 'orientation', 'posX', 'posY') # TODO: keep in mind to change the factor_names to lat_names based on the original script
    factor_sizes = np.array([1, 3, 6, 40, 32, 32])
    img_size = (1, 64, 64)
    background_color = datasets.COLOUR_BLACK
    factor_values = {
        'posX':
        np.array([
            0., 0.03225806, 0.06451613, 0.09677419, 0.12903226, 0.16129032,
            0.19354839, 0.22580645, 0.25806452, 0.29032258, 0.32258065,
            0.35483871, 0.38709677, 0.41935484, 0.4516129, 0.48387097,
            0.51612903, 0.5483871, 0.58064516, 0.61290323, 0.64516129,
            0.67741935, 0.70967742, 0.74193548, 0.77419355, 0.80645161,
            0.83870968, 0.87096774, 0.90322581, 0.93548387, 0.96774194, 1.
        ]),
        'posY':
        np.array([
            0., 0.03225806, 0.06451613, 0.09677419, 0.12903226, 0.16129032,
            0.19354839, 0.22580645, 0.25806452, 0.29032258, 0.32258065,
            0.35483871, 0.38709677, 0.41935484, 0.4516129, 0.48387097,
            0.51612903, 0.5483871, 0.58064516, 0.61290323, 0.64516129,
            0.67741935, 0.70967742, 0.74193548, 0.77419355, 0.80645161,
            0.83870968, 0.87096774, 0.90322581, 0.93548387, 0.96774194, 1.
        ]),
        'scale':
        np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
        'orientation':
        np.array([
            0., 0.16110732, 0.32221463, 0.48332195, 0.64442926, 0.80553658,
            0.96664389, 1.12775121, 1.28885852, 1.44996584, 1.61107316,
            1.77218047, 1.93328779, 2.0943951, 2.25550242, 2.41660973,
            2.57771705, 2.73882436, 2.89993168, 3.061039, 3.22214631,
            3.38325363, 3.54436094, 3.70546826, 3.86657557, 4.02768289,
            4.1887902, 4.34989752, 4.51100484, 4.67211215, 4.83321947,
            4.99432678, 5.1554341, 5.31654141, 5.47764873, 5.63875604,
            5.79986336, 5.96097068, 6.12207799, 6.28318531
        ]),
        'shape':
        np.array([1., 2., 3.]),
        'color':
        np.array([1.])
    }

    def __init__(self,
                 selected_factors,
                 not_selected_factors_index_value,
                 root='data/dsprites/',
                 drop_color_factor=True, # color has one value only and it is useless
                 **kwargs):
        """Initialize the DSprites dataset.
        
        This method loads the dataset from a numpy file, extracts the images and latent values,
        and processes them according to the selected factors of variation. The dataset contains
        2D shapes with controlled variations in color, shape, scale, orientation, and position.
        
        Parameters
        ----------
        selected_factors : list of str
            List of factor names to include in the dataset. Must be a subset of 
            ('color', 'shape', 'scale', 'orientation', 'posX', 'posY').
            These factors will be used as the latent variables.
            
        not_selected_factors_index_value : dict
            Dictionary mapping non-selected factor names to the index of their fixed value.
            For example, {'color': 0} would fix the color factor to its first possible value.
            
        root : str, default='data/dsprites/'
            Root directory where the dataset will be downloaded and stored.
        
        drop_color_factor : bool, default=True
            If True, the color factor will be dropped from output of factor values in __getitem__.
            This is useful because the color factor has only one value in this dataset.
            
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
        

        self.drop_color_factor = drop_color_factor
        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip['imgs']
        self.factor_values = dataset_zip['latents_values']
        # self.lat_values = sklearn.preprocessing.minimax_scale(self.lat_values)

        self.selected_img_indices = self._get_selected_img_indices()
        self.imgs = self.imgs[self.selected_img_indices]
        self._process_factor_values()

        if self.subset < 1:
            n_samples = int(len(self.imgs) * self.subset)
            subset = np.random.choice(len(self.imgs), n_samples, replace=False)
            self.imgs = self.imgs[subset]
            self.factor_values = self.factor_values[subset]

    def download(self):
        """Download the dataset."""
        os.makedirs(self.root)
        subprocess.check_call([
            "curl", "-L",
            type(self).urls["train"], "--output", self.train_data
        ])

    @property
    def name(self):
        """Name of the dataset."""
        return 'dsprites'

    @property
    def kwargs(self):
        """Keyword arguments for the dataset."""
        return {
            'selected_factors': self.selected_factors,
            'not_selected_factors_index_value': self.not_selected_factors_index_value,
            'root': self.root,
            'drop_color_factor': self.drop_color_factor,
            'subset': self.subset
        }

    def __getitem__(self, idx):
        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        if self.drop_color_factor:
            return self.transforms(sample), self.factor_values[idx][1:]
        else:
            return self.transforms(sample), self.factor_values[idx]
