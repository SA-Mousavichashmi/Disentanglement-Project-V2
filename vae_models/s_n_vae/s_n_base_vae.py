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
from power_spherical import PowerSpherical
import abc


class S_N_VAE_base(abc.ABC, nn.Module):
    """
    Base VAE model for mixed topologies that combines R1 (normal) and S1 (power spherical) components.
    Supports latent spaces with mixed topologies like ['R1', 'S1', 'R1'].
    """
    
    def __init__(self, img_size, latent_factor_topologies, decoder_output_dist='bernoulli', **kwargs):
        """
        Base class for mixed topology VAE.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        latent_factor_topologies : list of str
            List specifying the topology of each latent factor. Each element should be
            either 'R1' or 'R' for normal distribution or 'S1' for power spherical distribution.
            Example: ['R1', 'S1', 'R1']
        decoder_output_dist : str
            Distribution type for decoder output. Default is 'bernoulli'.
        """
        super(S_N_VAE_base, self).__init__()

        # Normalize topology names: 'R' -> 'R1'
        self.latent_factor_topologies = [t if t != 'R' else 'R1' for t in latent_factor_topologies]
        self.latent_factor_num = len(self.latent_factor_topologies)
        
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.decoder_output_dist = decoder_output_dist
        
        # Calculate parameters for each factor type
        self.factor_params = []
        self.factor_latent_dims = []
        
        for topology in self.latent_factor_topologies:
            if topology == 'R1':
                self.factor_params.append(2)  # mean, logvar
                self.factor_latent_dims.append(1)  # 1D latent space
            elif topology == 'S1':
                self.factor_params.append(3)  # mu_x, mu_y, kappa
                self.factor_latent_dims.append(2)  # 2D latent space (circle)
            else:
                raise ValueError(f"Unsupported topology: {topology}")
        
        # Total parameters and latent dimensions
        self.total_encoder_params = sum(self.factor_params)
        self.total_latent_dim = sum(self.factor_latent_dims)
        
        self.encoder = None
        self.decoder = None

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

    def _parse_encoder_output(self, encoder_output):
        """
        Parse the encoder output into factor-specific parameters.
        
        Parameters
        ----------
        encoder_output : torch.Tensor
            Encoder output with shape (batch_size, total_encoder_params)
            
        Returns
        -------
        stats_qzx : torch.Tensor
            Reformatted stats with shape (batch_size, latent_factor_num, max_params)
            where max_params is padded to accommodate all factor types.
        """
        batch_size = encoder_output.shape[0]
        max_params = max(self.factor_params)
        
        # Create tensor to hold all factor parameters with padding
        stats_qzx = torch.zeros(batch_size, self.latent_factor_num, max_params, 
                               device=encoder_output.device, dtype=encoder_output.dtype)
        
        # Parse encoder output into factor-specific parameters
        start_idx = 0
        for i, (topology, n_params) in enumerate(zip(self.latent_factor_topologies, self.factor_params)):
            end_idx = start_idx + n_params
            factor_params = encoder_output[:, start_idx:end_idx]
            
            if topology == 'R1':
                # For R1: params are [mean, logvar]
                stats_qzx[:, i, :n_params] = factor_params
            elif topology == 'S1':
                # For S1: params are [mu_x, mu_y, kappa]
                # Normalize mu and ensure kappa is positive
                mu_raw = factor_params[:, :2]  # First 2 params are mu
                kappa_raw = factor_params[:, 2]  # Third param is kappa
                
                mu_normalized = F.normalize(mu_raw, p=2, dim=-1)
                kappa_positive = F.softplus(kappa_raw) + 1e-4
                
                stats_qzx[:, i, :2] = mu_normalized
                stats_qzx[:, i, 2] = kappa_positive
            
            start_idx = end_idx
            
        return stats_qzx

    def reparameterize(self, stats_qzx):
        """
        Reparameterization trick for mixed topologies.
        
        Parameters
        ----------
        stats_qzx : torch.Tensor
            Factor parameters with shape (batch_size, latent_factor_num, max_params)
            
        Returns
        -------
        dict
            Dictionary with 'samples_qzx': sampled latent representations
        """
        batch_size = stats_qzx.shape[0]
        latent_samples = []
        
        for i, topology in enumerate(self.latent_factor_topologies):
            factor_params = stats_qzx[:, i, :]
            
            if topology == 'R1':
                # Normal distribution reparameterization
                mean = factor_params[:, 0]
                logvar = factor_params[:, 1]
                
                if self.training:
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    sample = mean + std * eps
                else:
                    sample = mean
                
                latent_samples.append(sample.unsqueeze(-1))  # Shape: (batch_size, 1)
                
            elif topology == 'S1':
                # Power Spherical distribution reparameterization
                mu = factor_params[:, :2]
                kappa = factor_params[:, 2]
                
                if self.training:
                    q_z_x = PowerSpherical(mu, kappa)
                    sample = q_z_x.rsample()
                else:
                    sample = mu
                
                latent_samples.append(sample)  # Shape: (batch_size, 2)
        
        # Concatenate all samples
        samples_qzx = torch.cat(latent_samples, dim=-1)  # Shape: (batch_size, total_latent_dim)
        
        return {'samples_qzx': samples_qzx}

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        # Get encoder output
        encoder_output = self.encoder(x)
        if isinstance(encoder_output, dict):
            # If encoder returns dict, extract the parameter tensor
            # Different encoders might use different keys
            if 'stats_qzx' in encoder_output:
                raw_params = encoder_output['stats_qzx']
                # If already has the right shape, use as is; otherwise flatten
                if raw_params.dim() == 3:
                    # Already in (batch_size, factor_num, params) format
                    batch_size, factor_num, param_dim = raw_params.shape
                    raw_params = raw_params.view(batch_size, -1)
                elif raw_params.dim() == 2:
                    # In (batch_size, total_params) format - use as is
                    pass
                else:
                    raise ValueError(f"Unexpected encoder output shape: {raw_params.shape}")
            else:
                # Try other common keys
                for key in ['output', 'params', 'logits']:
                    if key in encoder_output:
                        raw_params = encoder_output[key]
                        break
                else:
                    raise ValueError("Could not find parameter tensor in encoder output")
        else:
            # Encoder returns tensor directly
            raw_params = encoder_output

        # Parse encoder output into factor-specific parameters
        stats_qzx = self._parse_encoder_output(raw_params)

        # Reparameterization trick
        samples_qzx = self.reparameterize(stats_qzx)['samples_qzx']

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

    def reconstruct(self, x, mode='mean'):
        """
        Reconstructs the input data x using the VAE model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        mode : str
            Mode for reconstruction. Options are 'mean' or 'sample'.
        """
        # Get encoder output and parse it
        encoder_output = self.encoder(x)
        if isinstance(encoder_output, dict):
            if 'stats_qzx' in encoder_output:
                raw_params = encoder_output['stats_qzx']
                if raw_params.dim() == 3:
                    batch_size, factor_num, param_dim = raw_params.shape
                    raw_params = raw_params.view(batch_size, -1)
            else:
                for key in ['output', 'params', 'logits']:
                    if key in encoder_output:
                        raw_params = encoder_output[key]
                        break
        else:
            raw_params = encoder_output
            
        stats_qzx = self._parse_encoder_output(raw_params)

        if mode == 'mean':
            # Use deterministic reconstruction
            batch_size = stats_qzx.shape[0]
            latent_samples = []
            
            for i, topology in enumerate(self.latent_factor_topologies):
                factor_params = stats_qzx[:, i, :]
                
                if topology == 'R1':
                    # Use mean directly
                    mean = factor_params[:, 0]
                    latent_samples.append(mean.unsqueeze(-1))
                elif topology == 'S1':
                    # Use normalized mu directly
                    mu = factor_params[:, :2]
                    latent_samples.append(mu)
            
            samples_qzx = torch.cat(latent_samples, dim=-1)
            
        elif mode == 'sample':
            # Use stochastic reconstruction
            samples_qzx = self.reparameterize(stats_qzx)['samples_qzx']
        else:
            raise ValueError(f"Unknown reconstruction mode: {mode}")

        reconstructions = self.decoder(samples_qzx)['reconstructions']
        return reconstructions

    def sample_qzx(self, x, type='stochastic'):
        """
        Returns a sample z from the latent distribution q(z|x).

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        type : str
            Type of sampling to perform. Options are 'stochastic' or 'deterministic'.
        """
        # Get encoder output and parse it
        encoder_output = self.encoder(x)
        if isinstance(encoder_output, dict):
            if 'stats_qzx' in encoder_output:
                raw_params = encoder_output['stats_qzx']
                if raw_params.dim() == 3:
                    batch_size, factor_num, param_dim = raw_params.shape
                    raw_params = raw_params.view(batch_size, -1)
            else:
                for key in ['output', 'params', 'logits']:
                    if key in encoder_output:
                        raw_params = encoder_output[key]
                        break
        else:
            raw_params = encoder_output
            
        stats_qzx = self._parse_encoder_output(raw_params)

        if type == 'stochastic':
            samples_qzx = self.reparameterize(stats_qzx)['samples_qzx']
        elif type == 'deterministic':
            # Use means/modes
            batch_size = stats_qzx.shape[0]
            latent_samples = []
            
            for i, topology in enumerate(self.latent_factor_topologies):
                factor_params = stats_qzx[:, i, :]
                
                if topology == 'R1':
                    mean = factor_params[:, 0]
                    latent_samples.append(mean.unsqueeze(-1))
                elif topology == 'S1':
                    mu = factor_params[:, :2]
                    latent_samples.append(mu)
            
            samples_qzx = torch.cat(latent_samples, dim=-1)
        else:
            raise ValueError(f"Unknown sampling type: {type}")
        
        return samples_qzx

    def generate(self, num_samples, device):
        """
        Generates new images by sampling from the prior distributions.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        device : torch.device
            Device to perform computations on.
        """
        latent_samples = []
        
        for topology in self.latent_factor_topologies:
            if topology == 'R1':
                # Sample from standard normal
                sample = torch.randn(num_samples, 1, device=device)
                latent_samples.append(sample)
            elif topology == 'S1':
                # Sample uniformly from the circle
                angles = torch.rand(num_samples, 1, device=device) * 2 * torch.pi
                sample = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
                latent_samples.append(sample)
        
        # Concatenate all samples
        z = torch.cat(latent_samples, dim=-1)
        
        # Decode the latent samples to generate images
        generated_images = self.decoder(z)['reconstructions']
        
        return generated_images

    def reconstruct_latents(self, latent_z):
        """
        Reconstructs the input latent vector latent_z using the decoder.

        Parameters
        ----------
        latent_z : torch.Tensor
            Latent vector. Shape (batch_size, total_latent_dim)
        """
        reconstructions = self.decoder(latent_z)['reconstructions']
        return reconstructions

    def get_representations(self, x, is_deterministic=False):
        """
        Returns the latent representation of the input data x.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        is_deterministic : bool
            If True, returns the deterministic mean/mode of the latent distribution.
            If False, returns a stochastic sample using the reparameterization trick.
        """
        if is_deterministic:
            return self.sample_qzx(x, type='deterministic')
        else:
            return self.sample_qzx(x, type='stochastic')
