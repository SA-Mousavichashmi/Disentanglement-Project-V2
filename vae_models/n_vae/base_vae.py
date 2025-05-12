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
import abc


class BaseVAE(abc.ABC, nn.Module):
    """
    Base VAE model that contains common functionality for all VAE models.
    """
    def __init__(self, img_size, latent_dim=10, decoder_output_dist='bernoulli', **kwargs):
        """
        Base class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_dim : int
            Dimensionality of latent space.
        decoder_output_dist : str
            Distribution type for decoder output. Default is 'bernoulli'.
        """
        super(BaseVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.dist_nparams = 2
        self.encoder = None
        self.decoder = None
        self.decoder_output_dist = decoder_output_dist
    
    @property
    @abc.abstractmethod
    def name(self):
        """A unique name for the model, to be implemented by subclasses."""
        pass

    @property
    @abc.abstractmethod
    def model_kwargs(self):
        """Returns the keyword arguments for the model."""
        pass

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
        mean, logvar = stats_qzx.unbind(-1)

        # Reparameterization trick
        samples_qzx = self.reparameterize(mean, logvar)['samples_qzx']

        # Decode the latent samples
        reconstructions = self.decoder(samples_qzx)['reconstructions']

        return {
            'reconstructions': reconstructions, 
            'stats_qzx': stats_qzx, 
            'samples_qzx': samples_qzx,
            }

    def reset_parameters(self):
        """Reset parameters using weight_init."""
        self.apply(utils.initialization.weights_init)
    
    def reconstruct(self, x, mode):
        """
        Reconstructs the input data x using the VAE model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        mode : str
            Mode for reconstruction. Options are 'mean' or 'sample'.
        """
        stats_qzx = self.encoder(x)['stats_qzx']
        mean, logvar = stats_qzx.unbind(-1)

        if mode == 'mean':
            reconstructions = self.decoder(mean)['reconstructions']
        elif mode == 'sample':
            samples_qzx = self.reparameterize(mean, logvar)['samples_qzx']
            reconstructions = self.decoder(samples_qzx)['reconstructions']
        else:
            raise ValueError(f"Unknown reconstruction mode: {mode}")

        return reconstructions

    def reconstruct_latents(self, latent_z):
        """
        Reconstructs the input latent vector latent_z using the decoder.

        Parameters
        ----------
        latent_z : torch.Tensor
            Latent vector. Shape (batch_size, latent_dim)
        """
        reconstructions = self.decoder(latent_z)['reconstructions']
        return reconstructions


    def sample_qzx(self, x, type='stochastic'):
        """
        Returns a sample z from the latent distribution q(z|x) based on the type of sampling (stochastic or deterministic).

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        type : str
            Type of sampling to perform. Options are 'stochastic' or 'deterministic'.
        """
        stats_qzx = self.encoder(x)['stats_qzx']
        mean, logvar = stats_qzx.unbind(-1)

        if type == 'stochastic':
            samples_qzx = self.reparameterize(mean, logvar)['samples_qzx']
        elif type == 'deterministic':
            samples_qzx = mean
        else:
            raise ValueError(f"Unknown sampling type: {type}")
        
        return samples_qzx

    def generate(self, num_samples, device):
        """
        Generates new images by sampling from the prior distribution.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        device : torch.device
            Device to perform computations on.
        """
        # Sample latent vectors from the prior distribution (standard normal)
        z = torch.randn(num_samples, self.latent_dim).to(device)

        # Decode the latent samples to generate images
        generated_images = self.decoder(z)['reconstructions']

        return generated_images
    
    def get_representations(self, x, is_deterministic=False):
        """
        Returns the latent representation of the input data x.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        is_deterministic : bool
            If True, returns the deterministic mean of the latent distribution.
            If False, returns a stochastic sample using the reparameterization trick.
        """
        stats_qzx = self.encoder(x)['stats_qzx']
        mean, logvar = stats_qzx.unbind(-1)

        if is_deterministic:
            z = mean
        else:
            z = self.reparameterize(mean, logvar)['samples_qzx']

        return z
