# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from abc import ABC, abstractmethod


class BaseDecoder(nn.Module, ABC):
    """Base class for all decoders.    This class defines the common interface and functionality that all decoder
    implementations should follow.
    """

    def __init__(self, img_size, latent_dim=10, output_dist="bernoulli", use_batchnorm=False):
        """Initialize the base decoder.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        
        latent_dim : int
            Dimensionality of latent input.
            
        output_dist : str
            Type of output distribution. Either "bernoulli" or "gaussian".
            
        use_batchnorm : bool
            Whether to use batch normalization layers.
        """
        super(BaseDecoder, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.output_dist = output_dist
        self.use_batchnorm = use_batchnorm
        
        if output_dist not in ["bernoulli", "gaussian"]:
            raise ValueError(f"Output distribution {output_dist} not supported. Use 'bernoulli' or 'gaussian'.")

    @abstractmethod
    def decode(self, z):
        """Decode latent representations.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representation with shape [batch_size, latent_dim]
            
        Returns
        -------
        torch.Tensor
            Raw decoder output before applying activation function
        """
        pass
    
    def set_output_dist(self, x):
        """Process the decoder output based on output distribution.
        
        Parameters
        ----------
        x : torch.Tensor
            Raw decoder output
            
        Returns
        -------
        torch.Tensor
            Processed output. If output_dist is "bernoulli", applies sigmoid.
            If output_dist is "gaussian", returns raw values.
        """
        if self.output_dist == "bernoulli":
            return torch.sigmoid(x)
        elif self.output_dist == "gaussian":
            return x
            
    def forward(self, z):
        """Forward pass through the decoder.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representation with shape [batch_size, latent_dim]
            
        Returns
        -------
        dict
            Dictionary with key 'reconstructions' containing the decoded output
        """
        x = self.decode(z)
        x = self.set_output_dist(x)
        return {'reconstructions': x}