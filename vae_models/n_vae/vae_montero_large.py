# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, optim
from torch.nn import functional as F

from ..encoder.montero_large import Encoder
from ..decoder.montero_large import Decoder
from .base_vae import BaseVAE


class Model(BaseVAE):
    def __init__(self, img_size, latent_dim=10, decoder_output_dist='bernoulli', use_batchnorm=False, **kwargs):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_dim : int
            Dimensionality of latent space.
        decoder_output_dist : str
            Distribution type for decoder output. Default is 'bernoulli'.
        use_batchnorm : bool
            Whether to use batch normalization in encoder and decoder.
        """
        super(Model, self).__init__(img_size=img_size, latent_dim=latent_dim, 
                                   decoder_output_dist=decoder_output_dist,
                                   use_batchnorm=use_batchnorm, **kwargs)

        self.validate_img_size([[64, 64]])

        self.encoder = Encoder(
            img_size, self.latent_dim, dist_nparams=self.dist_nparams, use_batchnorm=use_batchnorm)
        self.decoder = Decoder(
            img_size, self.latent_dim, output_dist=decoder_output_dist, use_batchnorm=use_batchnorm)
        self.model_name = 'vae_montero_large'
        self.reset_parameters()

    @property
    def name(self):
        return 'vae_montero_large'

    @property
    def kwargs(self):
        return {
            'img_size': self.img_size,
            'latent_dim': self.latent_dim,
            'decoder_output_dist': self.decoder_output_dist,
            'use_batchnorm': self.use_batchnorm
        }
