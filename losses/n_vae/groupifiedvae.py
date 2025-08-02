# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from collections import OrderedDict

from .. import baseloss
from .. import select


class GroupifiedVAELoss(baseloss.BaseLoss):
    """
    Compute the Groupified VAE loss as proposed in "Towards Building a Group-based Unsupervised 
    Representation Disentanglement Framework" (Yang et al., ICLR 2022).
    
    This loss adds group theory constraints (isomorphism loss) to a base VAE loss function.
    The group constraints enforce cyclic group properties on meaningful latent dimensions.

    Parameters
    ----------
    base_loss_name : str
        Name of the base loss function (e.g., 'betavae', 'factorvae', etc.)
    
    base_loss_kwargs : dict
        Keyword arguments for the base loss function
    
    weight : float
        Weight for the group theory loss term
    
    action_scale : float, optional
        Scale for group actions in latent space. Default: 1.0
    
    N : int, optional
        Parameter for complex representation (periodicity). Default: 10
    
    kl_threshold : float, optional  
        KL divergence threshold for identifying meaningful dimensions. Default: 30.0
    
    fst_iter : int, optional
        First iteration to start applying group constraints. Default: 5000
        
    check_dims_freq : int, optional
        Frequency (in iterations) to check meaningful dimensions. Default: 200
        
    device : str, optional
        Device to run computations on. Default: 'cuda'
    
    schedulers_kwargs : list of dict, optional
        List of dictionaries containing scheduler configurations for parameters like 'weight'.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Yang, Tao, et al. "Towards building a group-based unsupervised representation 
        disentanglement framework." ICLR 2022.
    """

    def __init__(self, 
                 base_loss_name,
                 base_loss_kwargs,
                 weight,
                 action_scale=1.0,
                 N=10,
                 kl_threshold=30.0,
                 fst_iter=5000,
                 check_dims_freq=200,
                 device='cuda',
                 schedulers_kwargs=None,
                 **kwargs):
        
        super().__init__(mode="optimizes_internally", schedulers_kwargs=schedulers_kwargs, **kwargs)
        
        # Initialize schedulers for weight parameter
        if self.schedulers:
            if 'weight' in self.schedulers:
                weight = self.schedulers['weight'].initial_value
        
        self.base_loss_name = base_loss_name
        self.base_loss_kwargs = base_loss_kwargs
        self.weight = weight
        self.action_scale = action_scale
        self.N = N
        self.kl_threshold = kl_threshold
        self.fst_iter = fst_iter
        self.check_dims_freq = check_dims_freq
        self.device = device
        
        # Initialize base loss function
        self.base_loss_f = select(
            name=self.base_loss_name,
            **self.base_loss_kwargs,
            device=device
        )
        
        # Loss function for group constraints
        self.loss_func = nn.MSELoss()
        
        # State for meaningful dimensions
        self.mean_dims = [0, 1, 2, 3, 4, 5]  # Default dimensions
        self.current_iter = 0

    @property
    def name(self):
        return f'groupifiedvae_{self.base_loss_name}'

    @property
    def kwargs(self):
        kwargs_dict = {
            'base_loss_name': self.base_loss_name,
            'base_loss_kwargs': self.base_loss_kwargs,
            'weight': self.weight,
            'action_scale': self.action_scale,
            'N': self.N,
            'kl_threshold': self.kl_threshold,
            'fst_iter': self.fst_iter,
            'check_dims_freq': self.check_dims_freq,
            'device': self.device,
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
        
        # Save base loss state
        state['base_loss_state_dict'] = self.base_loss_f.state_dict()
        state['current_iter'] = self.current_iter
        state['mean_dims'] = self.mean_dims
        
        # Save scheduler states
        if self.schedulers:
            state['scheduler_states'] = {}
            for param_name, scheduler in self.schedulers.items():
                state['scheduler_states'][param_name] = scheduler.state_dict()
        
        return state

    def load_state_dict(self, state_dict):
        # Load base loss state
        if 'base_loss_state_dict' in state_dict:
            self.base_loss_f.load_state_dict(state_dict['base_loss_state_dict'])
        
        if 'current_iter' in state_dict:
            self.current_iter = state_dict['current_iter']
        
        if 'mean_dims' in state_dict:
            self.mean_dims = state_dict['mean_dims']
        
        # Load scheduler states
        if 'scheduler_states' in state_dict and self.schedulers:
            for param_name, scheduler_state in state_dict['scheduler_states'].items():
                if param_name in self.schedulers:
                    self.schedulers[param_name].load_state_dict(scheduler_state)

    # Group theory helper functions
    def complexfy(self, model, z):
        """Convert latent representation to complex representation using sine/cosine."""
        z = z[:, :model.latent_dim]
        real = torch.sin(2 * np.pi * z / self.N)
        imag = torch.cos(2 * np.pi * z / self.N)
        cm_z = torch.cat([real, imag], dim=1)
        return cm_z

    def forward_action(self, model, z, action_dim):
        """Apply forward group action on latent space."""
        mu = z.clone()
        mu = mu[:, :model.latent_dim].clone()
        mu[:, action_dim] += self.action_scale
        cm_z = self.complexfy(model, mu)
        
        # Decode through model
        x_recon = model.decoder(cm_z) # Keep in mind I removed sigmoid activation here

        return cm_z, x_recon

    def backward_action(self, model, z, action_dim):
        """Apply backward group action on latent space."""
        mu = z.clone()
        mu = mu[:, :model.latent_dim].clone()
        mu[:, action_dim] -= self.action_scale
        cm_z = self.complexfy(model, mu)
        
        # Decode through model
        x_recon = model.decoder(cm_z) # Keep in mind I removed sigmoid activation here

        return cm_z, x_recon

    def action_order_v2(self, model, x, action_dim):
        """Apply action order constraint (group property: g^-1 * g = identity)."""
        # Encode input
        zx = model.encoder(x)
        
        fcm = self.complexfy(model, zx)
        
        # Forward then backward action
        cm_z1, x1 = self.forward_action(model, zx, action_dim)
        z1 = model.encoder(x1)
        cm_z2 = self.complexfy(model, z1)
        fcm1, xr1 = self.backward_action(model, z1, action_dim)
        
        # Backward then forward action  
        cm_z3, x2 = self.backward_action(model, zx, action_dim)
        z2 = model.encoder(x2)
        cm_z4 = self.complexfy(model, z2)
        fcm2, xr2 = self.forward_action(model, z2, action_dim)
        
        return fcm, fcm1, fcm2, cm_z1, cm_z2, cm_z3, cm_z4, xr1, xr2

    def abel_action(self, model, x, a, b):
        """Apply abelian group constraint (commutativity: a*b = b*a)."""
        # Encode input
        zx = model.encoder(x)
        
        # Apply actions a then b
        cm_z1, x1 = self.forward_action(model, zx, a)
        z1 = model.encoder(x1)
        cm_z2 = self.complexfy(model, z1)
        fcm1, xr1 = self.forward_action(model, z1, b)
        
        # Apply actions b then a
        cm_z3, x2 = self.forward_action(model, zx, b)
        z2 = model.encoder(x2)
        cm_z4 = self.complexfy(model, z2)
        fcm2, xr2 = self.forward_action(model, z2, a)
        
        return fcm1, fcm2, cm_z1, cm_z2, cm_z3, cm_z4, xr1, xr2

    def constrain_order(self, model, x, key, mean_dims):
        """Apply order constraint for a specific dimension."""
        fcm, fcm1, fcm2, cm_z1, cm_z2, cm_z3, cm_z4, x1, x2 = self.action_order_v2(model, x, key)
        
        loss = (self.loss_func(fcm1[:, mean_dims], fcm[:, mean_dims]) +
                self.loss_func(fcm2[:, mean_dims], fcm[:, mean_dims]) +
                self.loss_func(cm_z1[:, mean_dims], cm_z2[:, mean_dims]) +
                self.loss_func(cm_z3[:, mean_dims], cm_z4[:, mean_dims]))
        
        return loss

    def constrain_abel(self, model, x, a, b, mean_dims):
        """Apply abelian constraint for dimensions a and b."""
        fcm1, fcm2, cm_z1, cm_z2, cm_z3, cm_z4, x1, x2 = self.abel_action(model, x, a, b)
        
        loss = (self.loss_func(fcm1[:, mean_dims], fcm2[:, mean_dims]) +
                self.loss_func(cm_z1[:, mean_dims], cm_z2[:, mean_dims]) +
                self.loss_func(cm_z3[:, mean_dims], cm_z4[:, mean_dims]))
        
        return loss

    def group_constrains(self, model, x, mean_dims):
        """Compute total group constraints (abelian + order)."""
        # Abelian constraints for all pairs of meaningful dimensions
        abloss = 0
        for j, com in enumerate(itertools.combinations(mean_dims, 2)):
            if j == 0:
                abloss = self.constrain_abel(model, x, com[0], com[1], mean_dims)
            else:
                abloss += self.constrain_abel(model, x, com[0], com[1], mean_dims)
        
        # Order constraints for all meaningful dimensions
        orloss = 0
        for i, key in enumerate(mean_dims):
            if i == 0:
                orloss = self.constrain_order(model, x, key, mean_dims)
            else:
                orloss += self.constrain_order(model, x, key, mean_dims)
        
        return abloss + orloss

    def check_dims(self, model, x):
        """Check which dimensions are meaningful based on KL divergence threshold."""
        with torch.no_grad():
            out = model.encoder(x).squeeze()
            
            # Extract mean and logvar
            latent_dim = model.latent_dim
            
            mean = out[:, :latent_dim].detach().cpu().numpy()
            logvar = out[:, latent_dim:].detach().cpu().numpy()
        
        # Compute KL divergence for each dimension
        kl = 0.5 * np.sum(np.square(mean) + np.exp(logvar) - logvar - 1, 0)
        meaningful_dims = np.where(kl > self.kl_threshold)[0]
        
        return meaningful_dims

    def __call__(self, data, model, optimizer, **kwargs):
        """
        Compute Groupified VAE loss.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data batch
        model : torch.nn.Module
            VAE model 
        optimizer : torch.optim.Optimizer
            Model optimizer
        kwargs : dict
            Additional arguments
            
        Returns
        -------
        dict
            Dictionary containing loss and logging information
        """
        self.current_iter += 1
        
        # Update schedulers
        if self.schedulers:
            self.step_schedulers()
        
        # Forward pass through model to get reconstructions and latent stats
        model_output = model(data)
        reconstructions = model_output['reconstructions']
        stats_qzx = model_output['stats_qzx']
        
        # Compute base loss
        if self.base_loss_f.mode == "post_forward":
            base_loss_result = self.base_loss_f(data, reconstructions, stats_qzx, **kwargs)
            base_loss = base_loss_result['loss']
            base_log_data = base_loss_result.get('to_log', {})
        else:
            # For other modes, we need to call differently
            base_loss_result = self.base_loss_f(data, model, optimizer, **kwargs)
            base_loss = base_loss_result['loss'] 
            base_log_data = base_loss_result.get('to_log', {})
        
        # Initialize total loss with base loss
        total_loss = base_loss
        log_data = base_log_data.copy()
        
        # Apply group constraints if we're past the first iteration threshold
        if self.current_iter > self.fst_iter:
            # Check meaningful dimensions periodically
            if self.current_iter % self.check_dims_freq == 0:
                self.mean_dims = self.check_dims(model, data).tolist()
            
            # Apply group constraints if we have at least 2 meaningful dimensions
            if len(self.mean_dims) >= 2:
                group_loss = self.group_constrains(model, data, self.mean_dims)
                total_loss = total_loss + self.weight * group_loss
                
                # Add to logging
                log_data['group_loss'] = group_loss.item()
                log_data['meaningful_dims'] = len(self.mean_dims)
            else:
                log_data['group_loss'] = 0.0
                log_data['meaningful_dims'] = len(self.mean_dims)
        else:
            log_data['group_loss'] = 0.0
            log_data['meaningful_dims'] = 0
        
        # Add weight to logging (helpful for tracking scheduled changes)
        log_data['group_weight'] = self.weight
        
        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'loss': total_loss,
            'to_log': log_data
        }
