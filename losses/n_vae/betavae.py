# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .. import baseloss
from ..reconstruction import reconstruction_loss
from .kl_div import kl_normal_loss


class Loss(baseloss.BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence.

    log_kl_components : bool, optional
        Whether to log individual KL components.

    schedulers_kwargs : list of dict, optional
        List of dictionaries containing scheduler configurations for parameters like 'beta'.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """

    def __init__(self, beta=1.0, log_kl_components=False, **kwargs):

        ##### parameters that is compatible for scheduling #####
        self.beta = beta

        super().__init__(mode="post_forward", **kwargs)
        
        # Initialize schedulers using base class method
        if self.schedulers:
            if not (len(self.schedulers) == 1 and 'beta' in self.schedulers):
                raise ValueError(f"Invalid scheduler configuration. Beta-VAE expects exactly one scheduler for 'beta', "
                                 f"but found {len(self.schedulers)} for: {list(self.schedulers.keys())}")
            
            beta = self.schedulers['beta'].get_value()
    
        self.log_kl_components = log_kl_components

    @property
    def name(self):
        return 'betavae'

    @property
    def kwargs(self):
        kwargs_dict = {
            'beta': self.beta,
            'log_kl_components': self.log_kl_components,
            'rec_dist': self.rec_dist,
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
        state = {}
        
        # Save scheduler states
        if self.schedulers:
            state['scheduler_states'] = {}
            for param_name, scheduler in self.schedulers.items():
                state['scheduler_states'][param_name] = scheduler.state_dict()
        
        return state if state else None
    
    def load_state_dict(self, state_dict):
        if state_dict is None:
            return
            
        # Load scheduler states
        if 'scheduler_states' in state_dict and self.schedulers:
            for param_name, scheduler_state in state_dict['scheduler_states'].items():
                if param_name in self.schedulers:
                    self.schedulers[param_name].load_state_dict(scheduler_state)

    def __call__(self, data, reconstructions, stats_qzx, **kwargs):   
            
        if isinstance(stats_qzx, torch.Tensor):
            stats_qzx = stats_qzx.unbind(-1)     

        # 1. Calculate all values first
        rec_loss = reconstruction_loss(data, reconstructions, distribution=self.rec_dist)
        kl_components = kl_normal_loss(*stats_qzx, return_components=True)
        kl_total = kl_components.sum()
        loss = rec_loss + self.beta * kl_total

        # 2. Initialize the dictionary
        log_data = {}

        # 3. Add items in the desired order
        log_data['loss'] = loss.item()
        log_data['rec_loss'] = rec_loss.item()
        log_data['kl_loss'] = kl_total.item()

        if self.log_kl_components:
            # Add individual components last (or wherever you prefer)
            for i, value in enumerate(kl_components):
                 log_data[f'kl_loss_{i}'] = value.item()
            # log_data['kl_components'] = kl_components.detach().cpu()

        return {'loss': loss, 'to_log': log_data} # TODO add separate loss related logs and other logs
