# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, optim
from torch.nn import functional as F

from ...encoder import select as select_encoder
from ...decoder import select as select_decoder
from .torodial_vae_base import Toroidal_VAE_Base


class Model(Toroidal_VAE_Base):
    def __init__(self, img_size, latent_factor_num=10, encoder_name='chen_mlp', decoder_name='chen_mlp', decoder_output_dist='bernoulli', **kwargs):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_factor_num : int
            Number of latent factors (dimensionality of the torus).
        encoder_name : str
            Name of encoder architecture to use.
        decoder_name : str
            Name of decoder architecture to use.
        decoder_output_dist : str
            Distribution type for decoder output. Default is 'bernoulli'.
        """
        super(Model, self).__init__(img_size=img_size, latent_factor_num=latent_factor_num, decoder_output_dist=decoder_output_dist, **kwargs)
        
        # self.validate_img_size([[32, 32], [64, 64]]) # Validation might depend on the specific encoder/decoder

        self.encoder = select_encoder(encoder_name)(
            img_size, self.latent_factor_num, dist_nparams=self.dist_nparams)
        self.decoder = select_decoder(decoder_name)(
            img_size, self.latent_factor_num * 2, output_dist=decoder_output_dist) # Corrected parameter name
        self.model_name = f'toroidal_vae_encoder-{encoder_name}_decoder-{decoder_name}'
        self.reset_parameters()

    @property
    def name(self):
        return f'toroidal_vae_encoder-{self.encoder_name}_decoder-{self.decoder_name}'
    
    @property
    def model_kwargs(self):
        return {
            'img_size': self.img_size,
            'latent_factor_num': self.latent_factor_num,
            'encoder_name': getattr(self, 'encoder_name', 'chen_mlp'),
            'decoder_name': getattr(self, 'decoder_name', 'chen_mlp'),
            'decoder_output_dist': self.decoder_output_dist
        }
