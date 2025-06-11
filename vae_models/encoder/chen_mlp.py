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
        r"""MLP Encoder of the model proposed in [1].

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
            [1] Chen et al. "Isolating Sources of Disentanglement in Variational Autoencoders"
        """
        super(Encoder, self).__init__(img_size, latent_dim, dist_nparams, use_batchnorm)

    def _build_network(self):
        self.hidden_dim = 1200
        self.input_dim = np.prod(self.img_size[-3:])
        # Layer parameters
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim, bias=not self.use_batchnorm)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim) if self.use_batchnorm else nn.Identity()
        
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.act = nn.ReLU(inplace=True)

    def _encode_features(self, x):
        h = x.view(-1, self.input_dim)
        h = self.act(self.bn1(self.fc1(h)))
        h = self.act(self.fc2(h))
        return h
