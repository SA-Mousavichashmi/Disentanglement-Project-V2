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
    def __init__(self, error_type="mse", reduction="mean", **kwargs):
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
        # Validate and store reduction mode for reconstruction error
        if reduction not in ["mean", "sum"]:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
        self.reduction = reduction

    @property
    def _requires(self):
        return ['reconstructions', 'data_samples']

    @property
    def _mode(self):
        return "batch"

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
        torch.Tensor
            Batch reconstruction error.
        """
        if self.error_type == "mse":
            # Squared reconstruction error for the batch
            return F.mse_loss(reconstructions, data_samples, reduction=self.reduction)
        elif self.error_type == "ce":
            # Binary cross-entropy reconstruction error for the batch
            return F.binary_cross_entropy(
                reconstructions, data_samples, reduction=self.reduction
            )
