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

    def __init__(self, img_size, latent_dim=10, output_dist="bernoulli"):
        r"""MLP Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.
            
        output_dist : str
            Type of output distribution. Either "bernoulli" or "gaussian".

        References:
            [1] Chen et al. "Isolating Sources of Disentanglement in Variational Autoencoders"
        """
        super(Decoder, self).__init__(img_size, latent_dim, output_dist)

        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, np.prod(self.img_size))
        )

    def decode(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        return h.view(z.size(0), *self.img_size)
