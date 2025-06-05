# filepath: c:\Users\Amint\Documents\GitHub\Disentanglement-Project-V2\losses\s_n_vae\beta_s_n_vae.py

import torch

from .. import baseloss
from ..reconstruction import reconstruction_loss
from ..n_vae.kl_div import kl_normal_loss
from ..s_vae.kl_div import kl_power_spherical_uniform_loss


class BetaSNVAELoss(baseloss.BaseLoss):
    """
    Compute the Beta-S-N-VAE loss, which combines R (Normal) and S1 (Power Spherical)
    latent factor topologies.

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence. Defaults to 1.0.
    latent_factor_topologies : list of str
        A list specifying the topology of each latent factor. Each element should be
        either 'R1' for a normal distribution or 'S1' for a power spherical distribution.
        Example: ['R1', 'S1', 'R1']
    log_kl_components : bool, optional
        Whether to log individual KL components. Defaults to False.

    kwargs:
        Additional arguments for `BaseLoss`, e.g., `rec_dist`.
    """

    def __init__(self, beta=1.0, latent_factor_topologies=None, log_kl_components=False, **kwargs):
        super().__init__(mode="post_forward", **kwargs)
        self.name = 'beta_s_n_vae'
        self.beta = beta
        if latent_factor_topologies is None:
            raise ValueError("latent_factor_topologies must be provided.")
        self.latent_factor_topologies = latent_factor_topologies
        self.log_kl_components = log_kl_components

    @property
    def name(self):
        return 'beta_s_n_vae'

    @property
    def kwargs(self):
        return {
            'beta': self.beta,
            'latent_factor_topologies': self.latent_factor_topologies,
            'log_kl_components': self.log_kl_components,
            'rec_dist': getattr(self, 'rec_dist', None),
        }
    
    def state_dict(self):
    # No state to save for this loss function beyond what BaseLoss handles
        return 

    def load_state_dict(self, state_dict):
        # No state to load for this loss function beyond what BaseLoss handles
        return


    def __call__(self, data, reconstructions, stats_qzx, is_train, **kwargs):
        # stats_qzx has shape (batch_size, latent_factor_num, dist_nparams)
        # We need to unbind along latent_factor_num to process each factor individually
        latent_factors_dist_param = stats_qzx.unbind(1)  # List of length latent_factor_num

        if len(self.latent_factor_topologies) != len(latent_factors_dist_param):
            raise ValueError(
                f"Number of latent factor topologies ({len(self.latent_factor_topologies)}) "
                f"does not match number of latent factors in stats_qzx ({len(latent_factors_dist_param)})."
            )

        # 1. Calculate all values first
        rec_loss = reconstruction_loss(data, reconstructions, distribution=self.rec_dist)

        kl_components_list = []
        for i, topology in enumerate(self.latent_factor_topologies):
            factor_params = latent_factors_dist_param[i]
            if topology == 'R1':
                # For Normal distribution, params are (mean, logvar)
                # kl_normal_loss expects two tensors: mean and logvar
                kl_component = kl_normal_loss(factor_params[:, 0], factor_params[:, 1], return_components=False).sum()
            elif topology == 'S1':
                # For Power Spherical distribution, params are (concentration, location, scale)
                # kl_power_spherical_uniform_loss expects a single tensor for each factor
                kl_component = kl_power_spherical_uniform_loss(factor_params, return_components=False).sum()
            else:
                raise ValueError(f"Unknown latent factor topology: {topology}")
            kl_components_list.append(kl_component)

        kl_total = torch.stack(kl_components_list).sum()  # Scalar tensor
        loss = rec_loss + self.beta * kl_total

        # 2. Initialize the dictionary for logging
        log_data = {}

        # 3. Add items in the desired order
        log_data['loss'] = loss.item()
        log_data['rec_loss'] = rec_loss.item()
        log_data['kl_loss'] = kl_total.item()

        if self.log_kl_components:
            # Add individual components last
            for i, value in enumerate(torch.stack(kl_components_list)):
                log_data[f'kl_loss_{i}'] = value.item()

        return {'loss': loss, 'to_log': log_data}
