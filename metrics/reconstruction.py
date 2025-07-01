"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn.functional as F
from .basemetric import BaseMetric

class ReconstructionError(BaseMetric):
    def __init__(self, error_type="mse", **kwargs):
        """
        Initialize ReconstructionError metric.
        
        Parameters
        ----------
        error_type : str, optional
            Type of reconstruction error to compute. 
            Options: "mse" (mean squared error) or "ce" (cross entropy).
            Defaults to "mse".
        """
        super().__init__(**kwargs)
        if error_type not in ["mse", "ce"]:
            raise ValueError(f"error_type must be 'mse' or 'ce', got {error_type}")
        self.error_type = error_type

    @property
    def _requires(self):
        return ['reconstructions', 'data_samples']

    @property
    def _mode(self):
        return 'instance'

    def __call__(self, reconstructions, data_samples, **kwargs):
        """Compute reconstruction error.
        
        Parameters
        ----------
        reconstructions : torch.Tensor
            Reconstructed data. Shape: (batch_size, n_chan, height, width).
        data_samples : torch.Tensor  
            Original data samples. Shape: (batch_size, n_chan, height, width).
            
        Returns
        -------
        tuple of torch.Tensor
            Per-sample reconstruction errors.
        """
        if self.error_type == "mse":
            # Mean-squared reconstruction error
            return torch.mean(
                (reconstructions - data_samples).view(len(data_samples), -1)**2, 
                dim=-1).unbind(-1)
        elif self.error_type == "ce":
            # Binary cross-entropy reconstruction error
            # Flatten the tensors for per-sample computation
            batch_size = data_samples.size(0)
            reconstructions_flat = reconstructions.view(batch_size, -1)
            data_samples_flat = data_samples.view(batch_size, -1)
            
            # Compute binary cross entropy per sample
            # Use reduction='none' to get per-sample losses, then sum over features
            bce_per_sample = F.binary_cross_entropy(
                reconstructions_flat, data_samples_flat, reduction='none'
            ).sum(dim=-1)
            
            return bce_per_sample.unbind(-1)