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
        r"""Decoder of the model proposed utilized in [1].

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
        - 5 convolutional layers with spatial broadcasting
        - Uses spatial coordinate grids as input

        References:
            [1] Locatello et al. "Weakly-Supervised Disentanglement without Compromises" 
            arXiv preprint https://arxiv.org/abs/2002.02886.
        """
        super(Decoder, self).__init__(img_size, latent_dim, output_dist, use_batchnorm)

        # Layer parameters
        kernel_size = 5
        self.img_size = img_size

        # Convolutional layers
        cnn_kwargs = dict(stride=1, padding=2, bias=not self.use_batchnorm)
        self.conv1 = nn.Conv2d(latent_dim + 2, 64, kernel_size, **cnn_kwargs)
        self.bn1 = nn.BatchNorm2d(64) if self.use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size, **cnn_kwargs)
        self.bn2 = nn.BatchNorm2d(64) if self.use_batchnorm else nn.Identity()
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size, **cnn_kwargs)
        self.bn3 = nn.BatchNorm2d(64) if self.use_batchnorm else nn.Identity()
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size, **cnn_kwargs)
        self.bn4 = nn.BatchNorm2d(64) if self.use_batchnorm else nn.Identity()
        
        # Last layer keeps bias so pixels can shift freely; no BN here        
        self.conv5 = nn.Conv2d(64, self.img_size[0], kernel_size, stride=1, padding=2)
        
        # XY Mesh.
        x, y = np.meshgrid(
            np.linspace(-1, 1, self.img_size[-2]),
            np.linspace(-1, 1, self.img_size[-1]))
        x = x.reshape(self.img_size[-2], self.img_size[-2], 1)
        y = y.reshape(self.img_size[-1], self.img_size[-1], 1)
        self.xy_mesh = torch.from_numpy(np.concatenate((x,y), axis=-1)).to(torch.float).unsqueeze(0)

    def spatial_broadcast(self, z):
        if self.xy_mesh.device != z.device:
            self.xy_mesh = self.xy_mesh.to(z.device)
        z_sb = torch.tile(z, (1, np.prod(self.img_size[-2:])))
        z_sb = z_sb.reshape(z.size(0), *self.img_size[-2:], z.size(-1))
        return torch.cat((z_sb, torch.tile(self.xy_mesh, (z.size(0), 1, 1, 1))), dim=3).permute(0, 3, 2, 1)

    def decode(self, z):
        # Apply Spatial Broadcasting.
        x = self.spatial_broadcast(z)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        # Return raw outputs (no activation)
        return self.conv5(x)
