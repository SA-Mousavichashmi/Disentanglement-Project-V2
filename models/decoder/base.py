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
    """Base class for all decoders.

    This class defines the common interface and functionality that all decoder
    implementations should follow.
    """

    def __init__(self, img_size, latent_dim=10, output_type="bernoulli"):
        """Initialize the base decoder.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        
        latent_dim : int
            Dimensionality of latent input.
            
        output_type : str
            Type of output distribution. Either "bernoulli" or "gaussian".
        """
        super(BaseDecoder, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.output_type = output_type
        
        if output_type not in ["bernoulli", "gaussian"]:
            raise ValueError(f"Output type {output_type} not supported. Use 'bernoulli' or 'gaussian'.")

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
        """Process the decoder output based on output type.
        
        Parameters
        ----------
        x : torch.Tensor
            Raw decoder output
            
        Returns
        -------
        torch.Tensor
            Processed output. If output_type is "bernoulli", applies sigmoid.
            If output_type is "gaussian", returns raw values.
        """
        if self.output_type == "bernoulli":
            return torch.sigmoid(x)
        elif self.output_type == "gaussian":
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