# filepath: c:\Users\Amint\Documents\GitHub\Disentanglement-Project-V2\losses\s_n_vae\anneal_s_n_vae.py

import torch

from .. import baseloss
from ..reconstruction import reconstruction_loss
from ..n_vae.kl_div import kl_normal_loss
from ..s_vae.kl_div import kl_power_spherical_uniform_factor_wise


class AnnealSNVAELoss(baseloss.BaseLoss):
    """
    Compute the Annealed-S-N-VAE loss, which combines R (Normal) and S1 (Power Spherical)
    latent factor topologies with annealed capacity constraint.

    Parameters
    ----------
    C_init : float, optional
        Starting annealed capacity C. Defaults to 0.0.
    C_fin : float, optional
        Final annealed capacity C. Defaults to 5.0.
    gamma : float, optional
        Weight of the KL divergence term. Defaults to 100.0.
    anneal_steps : int, optional
        Number of training steps over which to anneal. Defaults to 100000.
    latent_factor_topologies : list of str
        A list specifying the topology of each latent factor. Each element should be
        either 'R1' for a normal distribution or 'S1' for a power spherical distribution.
        Example: ['R1', 'S1', 'R1']
    log_kl_components : bool, optional
        Whether to log individual KL components. Defaults to False.
    state_dict : dict, optional
        State dictionary to load. Defaults to None.

    kwargs:
        Additional arguments for `BaseLoss`, e.g., `rec_dist`.

    References
    ----------
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
    """

    def __init__(self, 
                 C_init=0.0, 
                 C_fin=5.0, 
                 gamma=100.0, 
                 anneal_steps=100000,
                 latent_factor_topologies=None, 
                 log_kl_components=False,
                 state_dict=None,
                 **kwargs):
        
        ## parameters that is compatible for scheduling
        self.gamma = gamma
        
        super().__init__(mode="post_forward", **kwargs)
        
        # Initialize schedulers using base class method
        if self.schedulers:
            if not (len(self.schedulers) == 1 and 'gamma' in self.schedulers):
                raise ValueError(f"Invalid scheduler configuration. Annealed-S-N-VAE expects exactly one scheduler for 'gamma', "
                                 f"but found {len(self.schedulers)} for: {list(self.schedulers.keys())}")
            
            gamma = self.schedulers['gamma'].initial_value
        
        self.n_train_steps = 0
        self.gamma = gamma
        self.C_init = C_init
        self.C_fin = C_fin
        self.anneal_steps = anneal_steps
        if latent_factor_topologies is None:
            raise ValueError("latent_factor_topologies must be provided.")
        self.latent_factor_topologies = latent_factor_topologies
        self.log_kl_components = log_kl_components

        if state_dict is not None:
            self.load_state_dict(state_dict)

    @property
    def name(self):
        return 'anneal_s_n_vae'

    @property
    def kwargs(self):
        kwargs_dict = {
            'C_init': self.C_init,
            'C_fin': self.C_fin,
            'gamma': self.gamma,
            'anneal_steps': self.anneal_steps,
            'latent_factor_topologies': self.latent_factor_topologies,
            'log_kl_components': self.log_kl_components,
            'rec_dist': getattr(self, 'rec_dist', None),
        }
        
        # Add scheduler configurations
        if self.schedulers:
            schedulers_kwargs = []
            for param_name, scheduler in self.schedulers.items():
                schedulers_kwargs.append({
                    'name': scheduler.name,
                    'param_name': param_name,
                    'kwargs': {**scheduler.kwargs}
                })
            kwargs_dict['schedulers_kwargs'] = schedulers_kwargs
        
        return kwargs_dict
    
    def state_dict(self):
        state = {'n_train_steps': self.n_train_steps}
        
        # Save scheduler states
        if self.schedulers:
            state['scheduler_states'] = {}
            for param_name, scheduler in self.schedulers.items():
                state['scheduler_states'][param_name] = scheduler.state_dict()
        
        return state

    def load_state_dict(self, state_dict):
        self.n_train_steps = state_dict['n_train_steps']
        
        # Load scheduler states
        if 'scheduler_states' in state_dict and self.schedulers:
            for param_name, scheduler_state in state_dict['scheduler_states'].items():
                if param_name in self.schedulers:
                    self.schedulers[param_name].load_state_dict(scheduler_state)

    def _linear_annealing(self, init, fin, step, annealing_steps):
        """Linear annealing of a parameter."""
        if annealing_steps == 0:
            return fin
        assert fin > init, "Final value must be greater than initial value"
        delta = fin - init
        annealed = min(init + delta * step / annealing_steps, fin)
        return annealed

    def __call__(self, data, reconstructions, stats_qzx, is_train, **kwargs):
        # stats_qzx has shape (batch_size, total_encoder_params)
        # We need to parse the flattened parameter tensor based on latent_factor_topologies
        
        # Calculate parameter counts for each factor type
        factor_params = []
        for topology in self.latent_factor_topologies:
            if topology == 'R1':
                factor_params.append(2)  # mean, logvar
            elif topology == 'S1':
                factor_params.append(3)  # mu_x, mu_y, kappa
            else:
                raise ValueError(f"Unknown latent factor topology: {topology}")
        
        # Verify the total parameter count matches
        total_expected_params = sum(factor_params)
        if stats_qzx.shape[1] != total_expected_params:
            raise ValueError(
                f"Expected {total_expected_params} parameters but got {stats_qzx.shape[1]} "
                f"for topologies {self.latent_factor_topologies}."
            )

        # 1. Calculate all values first
        rec_loss = reconstruction_loss(data, reconstructions, distribution=self.rec_dist)

        kl_components_list = []
        start_idx = 0
        
        for i, (topology, n_params) in enumerate(zip(self.latent_factor_topologies, factor_params)):
            end_idx = start_idx + n_params
            factor_params_tensor = stats_qzx[:, start_idx:end_idx]
            
            if topology == 'R1':
                # For Normal distribution, params are (mean, logvar)
                # kl_normal_loss expects two tensors: mean and logvar
                mean = factor_params_tensor[:, 0]
                logvar = factor_params_tensor[:, 1]
                kl_component = kl_normal_loss(mean, logvar, return_components=False).sum()
            elif topology == 'S1':
                # For Power Spherical distribution, params are (mu_x, mu_y, kappa)
                kl_component = kl_power_spherical_uniform_factor_wise(factor_params_tensor)
            else:
                raise ValueError(f"Unknown latent factor topology: {topology}")
            
            kl_components_list.append(kl_component)
            start_idx = end_idx

        kl_total = torch.stack(kl_components_list).sum()  # Scalar tensor

        # Calculate annealed capacity
        C = (self._linear_annealing(self.C_init, self.C_fin, self.n_train_steps,
                              self.anneal_steps) if is_train else self.C_fin)

        # Apply annealed loss formula: rec_loss + gamma * |kl_loss - C|
        loss = rec_loss + self.gamma * (kl_total - C).abs()

        # 2. Initialize the dictionary for logging
        log_data = {}

        # 3. Add items in the desired order
        log_data['loss'] = loss.item()
        log_data['rec_loss'] = rec_loss.item()
        log_data['kl_loss'] = kl_total.item()
        log_data['annealed_capacity'] = C

        if self.log_kl_components:
            # Add individual components last
            for i, value in enumerate(kl_components_list):
                topology = self.latent_factor_topologies[i]
                log_data[f'kl_loss_{i}_{topology}'] = value.item()

        # Update training step counter
        if is_train:
            self.n_train_steps += 1

        return {'loss': loss, 'to_log': log_data}