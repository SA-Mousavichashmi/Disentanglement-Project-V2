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
import numpy as np


class BaseVAE(abc.ABC, nn.Module):
    """
    Base VAE model that contains common functionality for all VAE models.
    """
    
    def __init__(self, img_size, latent_dim=10, encoder_output_dim=None, decoder_input_dim=None, 
                 decoder_output_dist='bernoulli', use_batchnorm=False, use_complexify_rep=False, complexify_N=10, **kwargs):
        """
        Base class which defines model and forward pass.

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
        use_complexify_rep : bool
            Whether to complexify the latent representations. Default: False.
        complexify_N : int
            Parameter for complex representation (periodicity). Default: 10. Only used when use_complexify_rep=True.
        """
        super(BaseVAE, self).__init__()

        self.latent_dim = latent_dim
        self._encoder_output_dim = encoder_output_dim
        self._decoder_input_dim = decoder_input_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.dist_nparams = 2
        self.encoder = None
        self.decoder = None
        self.decoder_output_dist = decoder_output_dist
        self.use_batchnorm = use_batchnorm
        self.use_complexify_rep = use_complexify_rep
        self.complexify_N = complexify_N
        self.latent_factor_topologies = ['R1'] * latent_dim
    
    @property
    def encoder_output_dim(self):
        """Returns the effective encoder output dimension, using encoder_output_dim if set, otherwise latent_dim."""
        return self._encoder_output_dim if self._encoder_output_dim is not None else self.latent_dim
    
    @property
    def decoder_input_dim(self):
        """Returns the effective decoder input dimension, accounting for complexification if enabled."""

        if self.use_complexify_rep:
            # If using complexification, the input dimension is doubled (real + imaginary parts)
            decoder_input_dim = self.latent_dim * 2
        else:
            decoder_input_dim = self._decoder_input_dim if self._decoder_input_dim is not None else self.latent_dim
        
        return decoder_input_dim
    
    @property
    @abc.abstractmethod
    def name(self):
        """A unique name for the model, to be implemented by subclasses."""
        pass

    @property
    @abc.abstractmethod
    def kwargs(self):
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

    def complexify(self, z):
        """
        Convert latent representation to complex representation using sine/cosine.
        
        This implements the group-based complexification from Groupified-VAE,
        where the latent representation is transformed to include periodicity.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representation. Shape (batch_size, latent_dim)
            
        Returns
        -------
        torch.Tensor
            Complexified representation. Shape (batch_size, latent_dim * 2) if use_complexify_rep=True, else (batch_size, latent_dim)
        """

        # Apply trigonometric transformation with periodicity complexify_N
        real = torch.sin(2 * np.pi * z / self.complexify_N)
        imag = torch.cos(2 * np.pi * z / self.complexify_N)
        
        # Concatenate real and imaginary parts
        cm_z = torch.cat([real, imag], dim=1)
        return cm_z

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

        # Apply complexification (cosine/sine transformation) if use_complexify_rep=True
        if self.use_complexify_rep:
            decoder_input = self.complexify(samples_qzx)
        else:
            decoder_input = samples_qzx

        # Decode the latent samples (complexified if use_complexify_rep=True)
        reconstructions = self.decoder(decoder_input)['reconstructions']

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
            latent_z = mean
        elif mode == 'sample':
            latent_z = self.reparameterize(mean, logvar)['samples_qzx']
        else:
            raise ValueError(f"Unknown reconstruction mode: {mode}")

        # Apply complexification
        if self.use_complexify_rep:
            decoder_input = self.complexify(latent_z)
        else:
            decoder_input = latent_z

        reconstructions = self.decoder(decoder_input)['reconstructions']

        return reconstructions

    def reconstruct_latents(self, latent_z):
        """
        Reconstructs the input latent vector latent_z using the decoder.

        Parameters
        ----------
        latent_z : torch.Tensor
            Latent vector. Shape (batch_size, latent_dim)
        """
        # Apply complexification
        if self.use_complexify_rep:
            decoder_input = self.complexify(latent_z)
        else:
            decoder_input = latent_z

        reconstructions = self.decoder(decoder_input)['reconstructions']
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

        # Apply complexification if using group-based representation
        if self.use_complexify_rep:
            decoder_input = self.complexify(z)
        else:
            decoder_input = z

        # Decode the latent samples to generate images
        generated_images = self.decoder(decoder_input)['reconstructions']

        return generated_images
    
    def get_representations(self, x, is_deterministic=False, use_complexify_rep=False):
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

        if use_complexify_rep:
            z = self.complexify(z)


        return z
