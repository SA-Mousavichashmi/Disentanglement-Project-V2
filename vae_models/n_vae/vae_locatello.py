# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, optim
from torch.nn import functional as F

from ..encoder.locatello import Encoder
from ..decoder.locatello import Decoder
from .base_vae import BaseVAE


class Model(BaseVAE):
    def __init__(self, img_size, latent_dim=10, encoder_decay=0., decoder_decay=0., decoder_output_dist='bernoulli', use_batchnorm=False, **kwargs):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_dim : int
            Dimensionality of latent space.
        encoder_decay : float
            Weight decay for encoder parameters.
        decoder_decay : float
            Weight decay for decoder parameters.
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
        self.model_name = 'vae_locatello'
        self.reset_parameters()
        if encoder_decay or decoder_decay:
            self.to_optim = [
                {'params': self.encoder.parameters(), 'weight_decay': encoder_decay}, 
                {'params': self.decoder.parameters(), 'weight_decay': decoder_decay}
            ]

    @property
    def name(self):
        return 'vae_locatello'

    @property
    def kwargs(self):
        return {
            'img_size': self.img_size,
            'latent_dim': self.latent_dim,
            'encoder_decay': getattr(self, 'encoder_decay', 0.),
            'decoder_decay': getattr(self, 'decoder_decay', 0.),
            'decoder_output_dist': self.decoder_output_dist,
            'use_batchnorm': self.use_batchnorm
        }
