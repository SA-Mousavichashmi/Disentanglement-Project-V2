# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, optim
from torch.nn import functional as F

import utils.initialization
from ..encoder.burgess import Encoder
from ..decoder.burgess import Decoder
from .base_vae import BaseVAE


class Model(BaseVAE):
    def __init__(self, img_size, latent_dim=10, encoder_output_dim=None, decoder_input_dim=None, 
                 decoder_output_dist='bernoulli', use_batchnorm=False, use_complexify_rep=False, complexify_N=10, **kwargs):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_dim : int
            Dimensionality of latent space.
        encoder_output_dim : int, optional
            Dimensionality of encoder output. If None, uses latent_dim.
        decoder_input_dim : int, optional
            Dimensionality of decoder input. If None, uses latent_dim.
        decoder_output_dist : str
            Distribution type for decoder output. Default is 'bernoulli'.
        use_batchnorm : bool
            Whether to use batch normalization in encoder and decoder.
        """
        super(Model, self).__init__(img_size=img_size, latent_dim=latent_dim, 
                                   encoder_output_dim=encoder_output_dim,
                                   decoder_input_dim=decoder_input_dim,
                                   decoder_output_dist=decoder_output_dist, 
                                   use_batchnorm=use_batchnorm,
                                   use_complexify_rep=use_complexify_rep,
                                   complexify_N=complexify_N,
                                   **kwargs)
        
        self.validate_img_size([[32, 32], [64, 64]])

        self.encoder = Encoder(
            img_size, self.encoder_output_dim, dist_nparams=self.dist_nparams, use_batchnorm=use_batchnorm)
        self.decoder = Decoder(
            img_size, self.decoder_input_dim, output_dist=decoder_output_dist, use_batchnorm=use_batchnorm)
        self.model_name = 'vae_burgess'
        self.reset_parameters()

    @property
    def name(self):
        return 'vae_burgess'

    @property
    def kwargs(self):
        kwargs_dict = {
            'img_size': self.img_size,
            'latent_dim': self.latent_dim,
            'decoder_output_dist': self.decoder_output_dist,
            'use_batchnorm': self.use_batchnorm,
            'use_complexify_rep': self.use_complexify_rep,
            'complexify_N': self.complexify_N
        }
        if self._encoder_output_dim is not None:
            kwargs_dict['encoder_output_dim'] = self._encoder_output_dim
        if self._decoder_input_dim is not None:
            kwargs_dict['decoder_input_dim'] = self._decoder_input_dim
        return kwargs_dict
