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
from power_spherical import PowerSpherical  # type: ignore


class Toroidal_VAE_Base(nn.Module):
    """
    Base S-VAE (with S^1 * ... * S^1 latent space, N-Torus latent space) using Power spherical distribution.
    """
    def __init__(self, img_size, latent_factor_num=10, **kwargs):
        """
        Base class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_factor_num : int
            Dimensionality of latent space.
        """
        super(Toroidal_VAE_Base, self).__init__()

        self.latent_factor_num = latent_factor_num
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.dist_nparams = 2 + 1  # mu + kappa
        self.model_name = 'toroidal_vae_base'
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

    def reparameterize(self, latent_factors_dist_param):
        """
        Samples from the Power Spherical distribution for each latent factor using the reparameterization trick.

        During training, it samples from the distribution q(z|x) defined by the encoder's output parameters (mu, kappa).
        During evaluation (self.training=False), it deterministically returns the mean (mu) of the distribution.

        Parameters
        ----------
        latent_factors_dist_param : list of torch.Tensor
            A list containing the parameters (mu and kappa) for the Power Spherical distribution
            for each latent factor. Each tensor in the list has shape (batch_size, dist_nparams),
            where dist_nparams is 3 (2 for mu, 1 for kappa). The list has length `self.latent_factor_num`.

        Returns
        -------
        dict
            A dictionary containing:
            - 'samples_qzx': torch.Tensor of shape (batch_size, latent_factor_num, 2)
              The sampled or mean latent vectors for each factor. Each factor's sample is on S^1 (a 2D vector).
        """

        factorized_latent_samples = []

        for i in range(self.latent_factor_num):
            latent_factor_mu = latent_factors_dist_param[i][:, :2] # mu is the first two parameters
            latent_factor_kappa = latent_factors_dist_param[i][:, 2] # kappa is the third parameter

            if self.training: # TODO we can omit the this flag, the necessity of this flag is to be checked
                latent_factor_q_z_x = PowerSpherical(latent_factor_mu, latent_factor_kappa)
                sample_qzx = latent_factor_q_z_x.rsample()
            else:
                # During evaluation, we use the mean of the distribution
                sample_qzx = latent_factor_mu

            factorized_latent_samples.append(sample_qzx)
        
        latent_samples = torch.stack(factorized_latent_samples, dim=1) # Shape: (batch_size, latent_factor_num, 2)
        
        # Flatten the last two dimensions to get the final shape (batch_size, latent_factor_num * 2)
        latent_samples = latent_samples.flatten(start_dim=1) # Shape: (batch_size, latent_factor_num * 2)

        return {'samples_qzx': latent_samples}


    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        encoder_output = self.encoder(x) # Assuming encoder returns a dict like {'stats_qzx': ...}
        stats_qzx_raw = encoder_output['stats_qzx'] # stats_qzx has the shape (batch_size, latent_factor_num, dist_nparams)

        # Extract mu and kappa
        mu_raw = stats_qzx_raw[:, :, :2]
        kappa_raw = stats_qzx_raw[:, :, 2] # Shape: (batch_size, latent_factor_num)

        # Normalize mu
        mu_normalized = F.normalize(mu_raw, p=2, dim=-1)

        # Ensure kappa is positive
        kappa_positive = F.softplus(kappa_raw) + 1e-4 # Add epsilon for numerical stability

        # Combine normalized mu and positive kappa
        # Need to unsqueeze kappa to concatenate along the last dimension
        stats_qzx = torch.cat([mu_normalized, kappa_positive.unsqueeze(-1)], dim=-1)

        latent_factors_dist_param = stats_qzx.unbind(1)

        # Reparameterization trick
        samples_qzx = self.reparameterize(latent_factors_dist_param)['samples_qzx']

        print('samples_qzx shape:', samples_qzx.shape) # Debugging line

        # Decode the latent samples
        reconstructions = self.decoder(samples_qzx)['reconstructions']

        return {
            'reconstructions': reconstructions,
            'stats_qzx': stats_qzx, # Return the processed stats
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

        if mode == 'mean':
            # Extract the mean (mu) directly from the encoder's output
            # stats_qzx has shape (batch_size, latent_factor_num, dist_nparams)
            # mu is the first 2 parameters of dist_nparams
            samples_qzx = stats_qzx[:, :, :2].flatten(start_dim=1)
        elif mode == 'sample':
            # Use the reparameterize method to get a sample (respecting self.training)
            latent_factors_dist_param = stats_qzx.unbind(1)
            samples_qzx = self.reparameterize(latent_factors_dist_param)['samples_qzx']
        else:
            raise ValueError(f"Unknown reconstruction mode: {mode}")

        reconstructions = self.decoder(samples_qzx)['reconstructions']

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
        latent_factors_dist_param = stats_qzx.unbind(1)

        if type == 'stochastic':
            samples_qzx = self.reparameterize(latent_factors_dist_param)['samples_qzx']
        elif type == 'deterministic':
            samples_qzx = stats_qzx[:, :, :2].flatten(start_dim=1)  # Extract the mean (mu) directly from the encoder's output
        else:
            raise ValueError(f"Unknown sampling type: {type}")
        
        return samples_qzx

    def generate(self, num_samples, device):
        """
        Generates new images by sampling from the prior distribution (uniform on the N-Torus).

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        device : torch.device
            Device to perform computations on.
        """
        # Sample angles uniformly from [0, 2*pi) for each factor
        angles = torch.rand(num_samples, self.latent_factor_num, device=device) * 2 * torch.pi

        # Convert angles to points on the unit circle (S^1)
        # z_factor = (cos(angle), sin(angle))
        z_unflattened = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        # z_unflattened shape: (num_samples, latent_factor_num, 2)

        # Flatten the latent factors into a single vector per sample
        z = z_unflattened.flatten(start_dim=1)
        # z shape: (num_samples, latent_factor_num * 2)

        # Decode the latent samples to generate images
        generated_images = self.decoder(z)['reconstructions']

        return generated_images