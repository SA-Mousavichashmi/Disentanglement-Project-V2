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
        assert selected_factors == 'all' or isinstance(selected_factors, (list, dict)), "selected_factors must be a list of strings, a dict of factors with value indices, or 'all'."

        factor_names = type(self).factor_names
        self.not_selected_factors_index_value = not_selected_factors_index_value

        if selected_factors == 'all':
            assert not_selected_factors_index_value is None, "not_selected_factors_index_value must be None when all factors are selected."
            self.not_selected_factor_names = []
            self.selected_factor_indices = None  # New attribute for dict-based filtering
        elif isinstance(selected_factors, list):
            assert all([x in factor_names for x in selected_factors]), "Some of the selected factors are not in the dataset."
            assert isinstance(not_selected_factors_index_value, dict), "not_selected_factors_index_value must be a dictionary."

            not_selected_factor_names = list(not_selected_factors_index_value.keys())
    
            assert all([x in factor_names for x in not_selected_factor_names]), "Some of the not selected factors are not in the dataset."
            assert len(not_selected_factor_names) + len(selected_factors) == len(factor_names), "The selected and not selected factors must be the same as the dataset factors."

            self.not_selected_factor_names = not_selected_factor_names
            self.selected_factor_indices = None
        elif isinstance(selected_factors, dict):
            # New dict format: {factor_name: [indices] or "all"}
            assert not_selected_factors_index_value is None, "not_selected_factors_index_value must be None when selected_factors is a dict."
            assert all([x in factor_names for x in selected_factors.keys()]), "Some of the selected factors are not in the dataset."
            
            # Validate values in the dict
            for factor, value in selected_factors.items():
                if value != 'all':
                    assert isinstance(value, list), f"Value for factor '{factor}' must be a list of indices or 'all'."
                    assert all(isinstance(idx, int) for idx in value), f"All indices for factor '{factor}' must be integers."
            
            self.not_selected_factor_names = []
            self.selected_factor_indices = selected_factors


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
        Get indices of images that match the specified values for non-selected factors or selected factor indices.
        
        This method filters the dataset based on the selected_factors configuration:
        1. If 'all': Returns all image indices without filtering
        2. If list: Filters images to have specific values for non-selected factors
        3. If dict: Filters images to have specific value indices for each selected factor
        
        Returns
        -------
        np.array
            Array of indices of images that match the filtering criteria.
        """
        # Assertions
        assert self.factor_values is not None, "Factor values must be set before calling this method."
        assert self.imgs is not None, "Images must be set before calling this method."

        # Start with all indices
        selected_indices = np.arange(len(self.imgs))

        if self.selected_factors == 'all':
            # If all factors are selected, we don't need to filter
            return selected_indices
        elif isinstance(self.selected_factors, list):
            # Original list-based filtering logic
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
        elif isinstance(self.selected_factors, dict):
            # New dict-based filtering logic
            for factor, indices in self.selected_factors.items():
                factor_idx = self.factor_names.index(factor)
                factor_values = self.factor_values[:, factor_idx]
                unique_values = np.sort(np.unique(factor_values))
                
                if indices != 'all':
                    # Filter to only include specified indices for this factor
                    valid_indices = [idx for idx in indices if 0 <= idx < len(unique_values)]
                    target_values = unique_values[valid_indices]
                    
                    # Create mask for images that have any of the target values
                    mask = np.isin(factor_values[selected_indices], target_values)
                    selected_indices = selected_indices[mask]
        
        return selected_indices

    def _process_factor_values(self):
        """
        Filter factor values to only include selected factors, removing all non-selected factors.

        This method updates self.factor_values so that it contains only the columns corresponding to the selected factors
        for the filtered images (those in self.selected_img_indices). As a result, when accessing dataset items, only the
        relevant factor values (i.e., those for selected factors) are returned. All non-selected factors are removed from
        the returned factor_values.

        The method handles three cases:
        1. If all factors are selected ('all'), factor_values contains all factors for the selected images.
        2. If specific factors are selected as a list, factor_values contains only those selected factors for the selected images.
        3. If factors are selected as a dict, factor_values contains only the dict keys (selected factors) for the selected images.
        """
        assert self.factor_values is not None, "Factor values must be set before calling this method."
        assert self.selected_img_indices is not None, "Selected image indices must be set before calling this method."
        
        if self.selected_factors == 'all':
            # Include all factors for the selected images
            self.factor_values = self.factor_values[self.selected_img_indices]
        elif isinstance(self.selected_factors, list):
            # Get indices of selected factors in the factor_names list
            factor_indices = [
                self.factor_names.index(factor) 
                for factor in self.selected_factors
            ]
            # Filter both by selected images and selected factors
            self.factor_values = self.factor_values[self.selected_img_indices][:, factor_indices]
        elif isinstance(self.selected_factors, dict):
            # Get indices of selected factors (dict keys) in the factor_names list
            factor_indices = [
                self.factor_names.index(factor) 
                for factor in self.selected_factors.keys()
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