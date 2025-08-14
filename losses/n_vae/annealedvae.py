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
    C_init : float, optional
        Starting annealed capacity C.

    C_fin : float, optional
        Final annealed capacity C.

    gamma : float, optional
        Weight of the KL divergence term.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

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
                 log_kl_components=False,
                 state_dict=None,
                 **kwargs):
        
        ## parameters that is compatible for scheduling
        self.gamma = gamma

        super().__init__(mode="post_forward",**kwargs)
        
        # Initialize schedulers using base class method
        if self.schedulers:
            if not (len(self.schedulers) == 1 and 'gamma' in self.schedulers):
                raise ValueError(f"Invalid scheduler configuration. Annealed-VAE expects exactly one scheduler for 'gamma', "
                                 f"but found {len(self.schedulers)} for: {list(self.schedulers.keys())}")
            
            gamma = self.schedulers['gamma'].initial_value

        self.n_train_steps = 0
        self.C_init = C_init
        self.C_fin = C_fin
        self.anneal_steps = anneal_steps
        self.log_kl_components = log_kl_components

        if state_dict is not None:
            self.load_state_dict(state_dict)

    @property
    def name(self):
        return 'annealedvae'

    @property
    def kwargs(self):
        kwargs_dict = {
            'C_init': self.C_init,
            'C_fin': self.C_fin,
            'gamma': self.gamma,
            'anneal_steps': self.anneal_steps,
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
        if isinstance(stats_qzx, torch.Tensor):
            stats_qzx = stats_qzx.unbind(-1)

        log_data = {}

        rec_loss = reconstruction_loss(data, reconstructions, distribution=self.rec_dist)
        log_data['rec_loss'] = rec_loss.item()

        kl_components = kl_normal_loss(*stats_qzx, return_components=True) # Renamed from kl_loss to kl_components
        
        if self.log_kl_components:
            log_data.update(
                {f'kl_loss_{i}': value.item() for i, value in enumerate(kl_components)})
            # log_data['kl_components'] = kl_components.detach().cpu() # Log the tensor directly

        kl_loss = kl_components.sum() # Sum after potential logging
        log_data['kl_loss'] = kl_loss.item()

        C = (self._linear_annealing(self.C_init, self.C_fin, self.n_train_steps,
                              self.anneal_steps) if is_train else self.C_fin)

        loss = rec_loss + self.gamma * (kl_loss - C).abs()
        log_data['loss'] = loss.item()

        if is_train:
            self.n_train_steps += 1

        return {'loss': loss, 'to_log': log_data}

