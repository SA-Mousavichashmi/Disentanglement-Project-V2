# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
from torch import nn

class BaseEncoder(nn.Module):
    """Base class for all encoder models.
    
    This class contains common code and functionality shared across all encoder
    implementations. Specific encoder architectures should inherit from this class
    and implement the _build_network and _encode_features methods.
    """

    def __init__(self, img_size, latent_dim=10, dist_nparams=2):
        """Initialize the base encoder.
        
        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
            
        latent_dim : int
            Dimensionality of latent output.
            
        dist_nparams : int
            number of distribution statistics to return
        """
        super(BaseEncoder, self).__init__()
        
        # Store parameters
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.dist_nparams = dist_nparams
        
        # Build the network (to be implemented by subclasses)
        self._build_network()
        
        # All encoders have a final layer for distribution statistics
        self._build_dist_stats_layer()
    
    def _build_network(self):
        """Build the core network architecture.
        
        This method should be implemented by each subclass to define
        their specific network architecture.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _build_dist_stats_layer(self):
        """Build the distribution statistics layer."""
        # Subclasses should define self.hidden_dim before this method is called
        if not hasattr(self, 'hidden_dim'):
            raise AttributeError("Encoder subclass must define 'hidden_dim' before calling _build_dist_stats_layer.")
        
        # Fully connected layer for distribution statistics
        self.dist_statistics = nn.Linear(self.hidden_dim, self.latent_dim * self.dist_nparams)
    
    def _encode_features(self, x):
        """Encode the input into a feature vector.
        
        This method should be implemented by each subclass to process the input
        through their specific network architecture and return a feature vector
        ready for the distribution statistics layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        torch.Tensor
            Feature vector.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def forward(self, x):
        """Forward pass through the encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        dict
            Dictionary containing distribution statistics.
        """
        # Get features from the specific encoder implementation
        features = self._encode_features(x)
        
        # Generate distribution statistics
        dist_statistics = self.dist_statistics(features)
        
        # Return in the standard format used by all encoders
        return {'stats_qzx': dist_statistics.view(-1, self.latent_dim, self.dist_nparams)}