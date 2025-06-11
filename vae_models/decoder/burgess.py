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
        r"""Decoder of the model proposed in [1].

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
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(Decoder, self).__init__(img_size, latent_dim, output_dist, use_batchnorm)

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim, bias=not self.use_batchnorm)
        self.bn_lin1 = nn.BatchNorm1d(hidden_dim) if self.use_batchnorm else nn.Identity()
        
        self.lin2 = nn.Linear(hidden_dim, hidden_dim, bias=not self.use_batchnorm)
        self.bn_lin2 = nn.BatchNorm1d(hidden_dim) if self.use_batchnorm else nn.Identity()
        
        self.lin3 = nn.Linear(hidden_dim, np.prod(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1, bias=not self.use_batchnorm)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.bn_64 = nn.BatchNorm2d(hid_channels) if self.use_batchnorm else nn.Identity()

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.bn1 = nn.BatchNorm2d(hid_channels) if self.use_batchnorm else nn.Identity()
        
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.bn2 = nn.BatchNorm2d(hid_channels) if self.use_batchnorm else nn.Identity()
          # Last layer keeps bias so pixels can shift freely; no BN here
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, stride=2, padding=1)

    def decode(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.bn_lin1(self.lin1(z)))
        x = torch.relu(self.bn_lin2(self.lin2(x)))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.bn_64(self.convT_64(x)))
        x = torch.relu(self.bn1(self.convT1(x)))
        x = torch.relu(self.bn2(self.convT2(x)))
        # Return raw outputs (no activation)
        return self.convT3(x)
