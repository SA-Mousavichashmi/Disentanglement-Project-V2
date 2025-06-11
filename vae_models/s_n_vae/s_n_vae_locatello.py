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
from .s_n_base_vae import S_N_VAE_base


class Model(S_N_VAE_base):
    def __init__(self, img_size, latent_factor_topologies=['R1', 'S1', 'R1'], encoder_decay=0., decoder_decay=0., decoder_output_dist='bernoulli', use_batchnorm=False, **kwargs):
        """
        Mixed topology VAE using Locatello encoder/decoder architecture.
        Combines R1 (normal) and S1 (power spherical) latent factors.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_factor_topologies : list of str
            List specifying the topology of each latent factor. Each element should be
            either 'R1' or 'R' for normal distribution or 'S1' for power spherical distribution.
            Default: ['R1', 'S1', 'R1']
        encoder_decay : float
            Weight decay for encoder parameters.
        decoder_decay : float
            Weight decay for decoder parameters.
        decoder_output_dist : str
            Distribution type for decoder output. Default is 'bernoulli'.
        use_batchnorm : bool
            Whether to use batch normalization in encoder and decoder.
        """
        super(Model, self).__init__(
            img_size=img_size, 
            latent_factor_topologies=latent_factor_topologies, 
            decoder_output_dist=decoder_output_dist,
            use_batchnorm=use_batchnorm, 
            **kwargs
        )

        self.validate_img_size([[64, 64]])        # Create encoder that outputs the correct number of parameters
        # The encoder needs to output total_encoder_params dimensions
        self.encoder = Encoder(
            img_size, 
            self.total_encoder_params,  # Total parameters for all factors
            dist_nparams=1,  # We handle the distribution parameters ourselves
            use_batchnorm=use_batchnorm
        )
        
        # Create decoder that accepts the total latent dimensionality
        self.decoder = Decoder(
            img_size, 
            self.total_latent_dim,  # Total latent dimensions
            output_dist=decoder_output_dist,
            use_batchnorm=use_batchnorm
        )
        
        self.model_name = 's_n_vae_locatello'
        self.encoder_decay = encoder_decay
        self.decoder_decay = decoder_decay
        
        self.reset_parameters()
        
        if encoder_decay or decoder_decay:
            self.to_optim = [
                {'params': self.encoder.parameters(), 'weight_decay': encoder_decay}, 
                {'params': self.decoder.parameters(), 'weight_decay': decoder_decay}
            ]

    @property
    def name(self):
        return 's_n_vae_locatello'    @property
    def kwargs(self):
        return {
            'img_size': self.img_size,
            'latent_factor_topologies': self.latent_factor_topologies,
            'encoder_decay': getattr(self, 'encoder_decay', 0.),
            'decoder_decay': getattr(self, 'decoder_decay', 0.),
            'decoder_output_dist': self.decoder_output_dist,
            'use_batchnorm': self.use_batchnorm
        }