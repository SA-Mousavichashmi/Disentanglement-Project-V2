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
        r"""Encoder of the model proposed in [1].

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
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).        """
        super(Encoder, self).__init__(img_size, latent_dim, dist_nparams, use_batchnorm)

    def _build_network(self):
        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.hidden_dim = hidden_dim
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1, bias=not self.use_batchnorm)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.bn1 = nn.BatchNorm2d(hid_channels) if self.use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.bn2 = nn.BatchNorm2d(hid_channels) if self.use_batchnorm else nn.Identity()
        
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.bn3 = nn.BatchNorm2d(hid_channels) if self.use_batchnorm else nn.Identity()

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.bn_64 = nn.BatchNorm2d(hid_channels) if self.use_batchnorm else nn.Identity()

        # Fully connected layers
        self.lin1 = nn.Linear(np.prod(self.reshape), hidden_dim, bias=not self.use_batchnorm)
        self.bn_lin1 = nn.BatchNorm1d(hidden_dim) if self.use_batchnorm else nn.Identity()
        
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

    def _encode_features(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.bn_64(self.conv_64(x)))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.bn_lin1(self.lin1(x)))
        x = torch.relu(self.lin2(x))
        
        return x
