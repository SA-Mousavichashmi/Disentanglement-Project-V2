# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
from torch import nn

from .base import BaseEncoder

class Encoder(BaseEncoder):

    def __init__(self, img_size, latent_dim=10, dist_nparams=2, use_batchnorm=False):
        r"""Large Encoder as utilised in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        dist_nparams : int
            number of distribution statistics to return

        use_batchnorm : bool
            Whether to use batch normalization layers.

        References:
            [1] Montero et al. "Lost in Latent Space: Disentangled Models and 
            the Challenge of Combinatorial Generalisation."
        """
        super(Encoder, self).__init__(img_size, latent_dim, dist_nparams, use_batchnorm)

    def _build_network(self):
        # Layer parameters
        kernel_size = 4
        n_chan = self.img_size[0]

        assert_str = "This architecture requires 64x64 inputs."
        assert self.img_size[-2] == self.img_size[-1] == 64, assert_str

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1, bias=not self.use_batchnorm)
        self.conv1 = nn.Conv2d(n_chan, 64, kernel_size, **cnn_kwargs)
        self.bn1 = nn.BatchNorm2d(64) if self.use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size, **cnn_kwargs)
        self.bn2 = nn.BatchNorm2d(64) if self.use_batchnorm else nn.Identity()
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size, **cnn_kwargs)
        self.bn3 = nn.BatchNorm2d(128) if self.use_batchnorm else nn.Identity()
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size, **cnn_kwargs)
        self.bn4 = nn.BatchNorm2d(128) if self.use_batchnorm else nn.Identity()
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size, **cnn_kwargs)
        self.bn5 = nn.BatchNorm2d(256) if self.use_batchnorm else nn.Identity()

        # Fully connected layers
        inp_width = int(64/(2**5))
        self.lin = nn.Linear(inp_width**2 * 256, 256)
        
        # Define hidden_dim for the distribution statistics layer
        self.hidden_dim = 256

    def _encode_features(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin(x))
        
        return x
