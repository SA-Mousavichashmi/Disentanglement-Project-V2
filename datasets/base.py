# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
import os

import torch
import torchvision
import numpy as np

class DisentangledDataset(torch.utils.data.Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self,
                 root,
                 selected_factors,
                 not_selected_factors_index_value,
                 transforms_list=[],
                 logger=logging.getLogger(__name__),
                 subset=1):
        
        # Assertions
        assert selected_factors == 'all' or isinstance(selected_factors, list), "selected_factors must be a list of strings or 'all.'"

        if selected_factors == 'all':
            assert not_selected_factors_index_value is None, "not_selected_factors_index_value must be None when all factors are selected."

        factor_names = type(self).factor_names
        self.not_selected_factors_index_value = not_selected_factors_index_value

        if isinstance(selected_factors, list):
            assert all([x in factor_names for x in selected_factors]), "Some of the selected factors are not in the dataset."
            assert isinstance(not_selected_factors_index_value, dict), "not_selected_factors_index_value must be a dictionary."

            not_selected_factor_names = list(not_selected_factors_index_value.keys())
    
            assert all([x in factor_names for x in not_selected_factor_names]), "Some of the not selected factors are not in the dataset."
            assert len(not_selected_factor_names) + len(selected_factors) == len(factor_names), "The selected and not selected factors must be the same as the dataset factors."

            self.not_selected_factor_names = not_selected_factor_names
        else:
            self.not_selected_factor_names = []


        self.root = root
        self.selected_factors = selected_factors
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = torchvision.transforms.Compose(transforms_list)
        self.logger = logger
        self.subset = subset
        self.factor_values = None
        self.imgs = None
        self.selected_img_indices = None

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.selected_img_indices)


    def _get_selected_img_indices(self):
        """
        Get indices of images that match the specified values for non-selected factors.
        
        This method filters the dataset to only include images that have specific values
        for factors that were not selected during initialization. For each non-selected
        factor, it keeps only images where that factor has the value specified in 
        the not_selected_factors_index_value dictionary. If all factors are selected
        ('all'), returns all image indices without filtering.
        
        Returns
        -------
        np.array
            Array of indices of images that match the specified values for non-selected factors.
        """
        # Assertions
        assert self.factor_values is not None, "Factor values must be set before calling this method."
        assert self.imgs is not None, "Images must be set before calling this method."

        # Start with all indices
        selected_indices = np.arange(len(self.imgs))

        if self.selected_factors == 'all':
            # If all factors are selected, we don't need to filter
            return selected_indices
          
        # For each non-selected factor with a specified fixed value, filter the indices
        for factor in self.not_selected_factor_names:
            if factor in self.not_selected_factors_index_value:
                factor_idx = self.factor_names.index(factor)
                value_idx = self.not_selected_factors_index_value[factor]
                
                # Get all unique values for this factor
                factor_values = self.factor_values[:, factor_idx]
                unique_values = np.sort(np.unique(factor_values))
                
                # Get the target value based on the specified index
                if 0 <= value_idx < len(unique_values):
                    target_value = unique_values[value_idx]
                    
                    # Filter indices to only include images with the target value
                    selected_indices = selected_indices[factor_values[selected_indices] == target_value]
        
        return selected_indices

    def _process_factor_values(self):
        """
        Filter factor values to only include selected factors, removing all non-selected factors.

        This method updates self.factor_values so that it contains only the columns corresponding to the selected factors
        for the filtered images (those in self.selected_img_indices). As a result, when accessing dataset items, only the
        relevant factor values (i.e., those for selected factors) are returned. All non-selected factors are removed from
        the returned factor_values.

        The method handles two cases:
        1. If all factors are selected ('all'), factor_values contains all factors for the selected images.
        2. If specific factors are selected, factor_values contains only those selected factors for the selected images.
        """
        assert self.factor_values is not None, "Factor values must be set before calling this method."
        assert self.selected_img_indices is not None, "Selected image indices must be set before calling this method."
        
        if self.selected_factors == 'all':
            # Include all factors for the selected images
            self.factor_values = self.factor_values[self.selected_img_indices]
        else:
            # Get indices of selected factors in the factor_names list
            factor_indices = [
                self.factor_names.index(factor) 
                for factor in self.selected_factors
            ]
            # Filter both by selected images and selected factors
            self.factor_values = self.factor_values[self.selected_img_indices][:, factor_indices]

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass

    @property
    @abc.abstractmethod
    def name(self):
        """Name of the dataset."""
        pass

    @property
    @abc.abstractmethod
    def kwargs(self):
        """Keyword arguments for the dataset."""
        pass