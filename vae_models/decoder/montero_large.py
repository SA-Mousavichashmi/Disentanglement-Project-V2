# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
from torch import nn

from .base import BaseDecoder


class Decoder(BaseDecoder):

    def __init__(self, img_size, latent_dim=10, output_dist="bernoulli", use_batchnorm=False):
        r"""Large Decoder as used in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.
            
        output_dist : str
            Type of output distribution. Either "bernoulli" or "gaussian".
            
        use_batchnorm : bool
            Whether to use batch normalization layers.

        Model Architecture (transposed for decoder)
        ------------
        - 5 convolutional layers with increasing channels, (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Montero et al. "Lost in Latent Space: Disentangled Models and 
            the Challenge of Combinatorial Generalization."
        """
        super(Decoder, self).__init__(img_size, latent_dim, output_dist, use_batchnorm)

        # Layer parameters
        kernel_size = 4
        self.img_size = img_size
        # Shape required to start transpose convs
        n_chan = self.img_size[0]
        inp_width = int(64/(2**5))
        self.reshape = (256, inp_width, inp_width)

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, 256, bias=not self.use_batchnorm)
        self.bn_lin1 = nn.BatchNorm1d(256) if self.use_batchnorm else nn.Identity()
        
        self.lin2 = nn.Linear(256, inp_width**2 * 256)

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1, bias=not self.use_batchnorm)
        self.convT1 = nn.ConvTranspose2d(256, 128, kernel_size, **cnn_kwargs)
        self.bn1 = nn.BatchNorm2d(128) if self.use_batchnorm else nn.Identity()
        
        self.convT2 = nn.ConvTranspose2d(128, 128, kernel_size, **cnn_kwargs)
        self.bn2 = nn.BatchNorm2d(128) if self.use_batchnorm else nn.Identity()
        
        self.convT3 = nn.ConvTranspose2d(128, 64, kernel_size, **cnn_kwargs)
        self.bn3 = nn.BatchNorm2d(64) if self.use_batchnorm else nn.Identity()
        
        self.convT4 = nn.ConvTranspose2d(64, 64, kernel_size, **cnn_kwargs)
        self.bn4 = nn.BatchNorm2d(64) if self.use_batchnorm else nn.Identity()
        
        # Last layer keeps bias so pixels can shift freely; no BN here        
        self.convT5 = nn.ConvTranspose2d(64, n_chan, kernel_size, stride=2, padding=1)

    def decode(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.bn_lin1(self.lin1(z)))
        x = torch.relu(self.lin2(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.bn1(self.convT1(x)))
        x = torch.relu(self.bn2(self.convT2(x)))
        x = torch.relu(self.bn3(self.convT3(x)))
        x = torch.relu(self.bn4(self.convT4(x)))
        # Return raw outputs (no activation)
        return self.convT5(x)
