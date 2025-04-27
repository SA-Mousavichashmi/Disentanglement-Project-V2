# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .. import baseloss
from ..reconstruction import reconstruction_loss
from .kl_div import kl_power_spherical_uniform_loss

class BetaToroidalVAELoss(baseloss.BaseLoss):
    """
    Compute the Beta-Toroidal-VAE loss. This is similar to Beta-VAE but uses
    the KL divergence for the Power Spherical distribution against a uniform prior on S^1.

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence. Defaults to 1.0.
    
    log_components : bool, optional
        Whether to log individual KL components. Defaults to False.

    kwargs:
        Additional arguments for `BaseLoss`, e.g., `rec_dist`.
    """

    def __init__(self, beta=1.0, log_kl_components=False, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.log_kl_components = log_kl_components

    def __call__(self, data, reconstructions, stats_qzx, is_train, **kwargs):   
        """
        Calculates the Beta-Toroidal-VAE loss.

        Parameters
        ----------
        data : torch.Tensor
            Input data batch.
        reconstructions : torch.Tensor
            Reconstructed data batch from the VAE.
        stats_qzx : torch.Tensor
            Tensor containing the parameters of the posterior distribution q(z|x) 
            output by the encoder. Shape: (batch_size, latent_factor_num, dist_nparams=3).
        is_train : bool
            Flag indicating whether the model is in training mode.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary containing:
            - 'loss': The total computed loss (scalar tensor).
            - 'to_log': A dictionary with components of the loss for logging purposes.
        """
        self._pre_call(is_train)

        # stats_qzx has shape (batch_size, latent_factor_num, dist_nparams)
        # kl_toroidal_loss expects a list of tensors, one for each factor
        latent_factors_dist_param = stats_qzx.unbind(1) # List of length latent_factor_num, each tensor shape (batch_size, dist_nparams)

        # 1. Calculate all values first
        rec_loss = reconstruction_loss(data, reconstructions, distribution=self.rec_dist)
        kl_components = kl_power_spherical_uniform_loss(latent_factors_dist_param, return_components=True) # Returns shape (latent_factor_num,)
        kl_total = kl_components.sum() # Scalar tensor
        loss = rec_loss + self.beta * kl_total

        # 2. Initialize the dictionary for logging
        log_data = {}

        # 3. Add items in the desired order
        log_data['loss'] = loss.item()
        log_data['rec_loss'] = rec_loss.item()
        log_data['kl_loss'] = kl_total.item()

        if self.log_kl_components:
            # Add individual components last
            # for i, value in enumerate(kl_components):
            #      log_data[f'kl_loss_{i}'] = value.item()
            log_data['kl_components'] = kl_components.cpu() # Log the tensor directly

        return {'loss': loss, 'to_log': log_data}
