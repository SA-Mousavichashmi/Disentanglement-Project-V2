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


class BaseVAE(nn.Module):
    """
    Base VAE model that contains common functionality for all VAE models.
    """
    def __init__(self, img_size, latent_dim=10, **kwargs):
        """
        Base class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_dim : int
            Dimensionality of latent space.
        """
        super(BaseVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.dist_nparams = 2
        self.model_name = 'base_vae'
        self.encoder = None
        self.decoder = None

    def validate_img_size(self, allowed_sizes):
        """
        Validates that the image size is supported by the model.

        Parameters
        ----------
        allowed_sizes : list of lists
            List of supported image dimensions, e.g. [[32, 32], [64, 64]]
        """
        if list(self.img_size[1:]) not in allowed_sizes:
            supported = ", ".join([f"(None, {h}, {w})" for h, w in allowed_sizes])
            raise RuntimeError(
                f"{self.img_size} sized images not supported. Only {supported} supported. "
                "Build your own architecture or reshape images!"
            )

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return {'samples_qzx': mean + std * eps}
        else:
            # Reconstruction mode
            return {'samples_qzx': mean}

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        stats_qzx = self.encoder(x)['stats_qzx']
        samples_qzx = self.reparameterize(*stats_qzx.unbind(-1))['samples_qzx']
        reconstructions = self.decoder(samples_qzx)['reconstructions']
        return {
            'reconstructions': reconstructions, 
            'stats_qzx': stats_qzx, 
            'samples_qzx': samples_qzx}

    def reset_parameters(self):
        """Reset parameters using weight_init."""
        self.apply(utils.initialization.weights_init)

    def sample_qzx(self, x):
        """
        Returns a sample z from the latent distribution q(z|x).

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        stats_qzx = self.encoder(x)['stats_qzx']
        samples_qzx = self.reparameterize(*stats_qzx.unbind(-1))['samples_qzx']
        return samples_qzx