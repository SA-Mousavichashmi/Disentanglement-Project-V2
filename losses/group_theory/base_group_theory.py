"""
Base class for Group Theory losses to eliminate code duplication between n_vae and s_n_vae implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from collections import OrderedDict

from ..baseloss import BaseLoss
from ..reconstruction import reconstruction_loss
from .. import select
from ..n_vae.kl_div import kl_normal_loss
from ..s_vae.kl_div import kl_power_spherical_uniform_factor_wise
from .base_utils import select_latent_components


class BaseGroupTheoryLoss(BaseLoss, ABC):
    """
    Base class for Group Theory losses that provides common functionality for both R^1 and mixed topology implementations.
    
    This class handles:
    - Common initialization logic
    - Base loss integration
    - Scheduler management
    - State persistence
    - Warm-up logic
    - Critic training for meaningful loss
    - Main training loop structure
    
    Subclasses need to implement topology-specific methods for:
    - Parameter generation
    - Group action application
    - Component selection utilities
    """
    
    def __init__(self,
                 base_loss_name,
                 base_loss_kwargs,
                 rec_dist,
                 device,
                 commutative_weight,
                 commutative_component_order,
                 commutative_comparison_dist,
                 meaningful_weight,
                 meaningful_component_order,
                 meaningful_transformation_order,
                 meaningful_critic_gradient_penalty_weight,
                 meaningful_critic_lr,
                 meaningful_n_critic,
                 deterministic_rep,
                 comp_latent_select_threshold=0,
                 base_loss_state_dict=None,
                 warm_up_steps=0,
                 schedulers_kwargs=None,
                 **kwargs):
        
        super().__init__(mode="optimizes_internally", 
                        rec_dist=rec_dist, 
                        schedulers_kwargs=schedulers_kwargs, 
                        **kwargs)
        
        # Initialize scheduler values if they exist
        if self.schedulers: 
            if 'commutative_weight' in self.schedulers:
                commutative_weight = self.schedulers['commutative_weight'].initial_value
            if 'meaningful_weight' in self.schedulers:
                meaningful_weight = self.schedulers['meaningful_weight'].initial_value

        # Base loss configuration
        self.base_loss_name = base_loss_name
        self.base_loss_kwargs = base_loss_kwargs
        self.base_loss_state_dict = base_loss_state_dict
        self.base_loss_f = select( 
            name=self.base_loss_name, 
            **self.base_loss_kwargs,
            state_dict=self.base_loss_state_dict,
            device=device
        )

        # Core parameters
        self.rec_dist = rec_dist
        self.device = device
        self.deterministic_rep = deterministic_rep
        
        # Group theory weights
        self.commutative_weight = commutative_weight
        self.meaningful_weight = meaningful_weight
        
        # Component orders
        self.commutative_component_order = commutative_component_order
        self.meaningful_component_order = meaningful_component_order
        
        # Validation
        if self.commutative_component_order <= 1:
            raise ValueError("commutative_component_order must be greater than 1 for the commutative loss calculation.")
             
        # Meaningful loss parameters
        self.meaningful_transformation_order = meaningful_transformation_order
        self.meaningful_critic_gradient_penalty_weight = meaningful_critic_gradient_penalty_weight
        self.meaningful_critic_lr = meaningful_critic_lr
        self.meaningful_n_critic = meaningful_n_critic

        # Critic components (initialized lazily)
        self.critic = None
        self.critic_optimizer = None

        # Comparison and selection parameters
        self.commutative_comparison_dist = commutative_comparison_dist
        if self.commutative_comparison_dist not in ['gaussian', 'bernoulli']:
            raise ValueError("commutative_comparison_dist must be either 'gaussian' or 'bernoulli'.")
        
        self.comp_latent_select_threshold = comp_latent_select_threshold
        if self.comp_latent_select_threshold < 0 or self.comp_latent_select_threshold >= 1:
            raise ValueError("comp_latent_select_threshold must be in the range [0, 1).")

        # Warm-up parameters
        self.warm_up_steps = warm_up_steps
        self.current_step = 0

        # Topology information (set from model)
        self.latent_factor_topologies = None

    @property
    @abstractmethod
    def name(self):
        """Return the name of the group theory loss variant."""
        pass
    
    @property
    @abstractmethod
    def topology_specific_kwargs(self):
        """Return topology-specific keyword arguments for serialization."""
        pass
    
    @property
    def kwargs(self):
        """Return all keyword arguments needed to reconstruct this loss instance."""
        kwargs_dict = {
            'base_loss_name': self.base_loss_name,
            'base_loss_kwargs': self.base_loss_kwargs,
            'rec_dist': self.rec_dist,
            'device': self.device,
            'commutative_weight': self.commutative_weight,
            'commutative_component_order': self.commutative_component_order,
            'commutative_comparison_dist': self.commutative_comparison_dist,
            'meaningful_weight': self.meaningful_weight,
            'meaningful_component_order': self.meaningful_component_order,
            'meaningful_transformation_order': self.meaningful_transformation_order,
            'meaningful_critic_gradient_penalty_weight': self.meaningful_critic_gradient_penalty_weight,
            'meaningful_critic_lr': self.meaningful_critic_lr,
            'meaningful_n_critic': self.meaningful_n_critic,
            'deterministic_rep': self.deterministic_rep,
            'comp_latent_select_threshold': self.comp_latent_select_threshold,
            'warm_up_steps': self.warm_up_steps,
            'current_step': self.current_step,
        }
        
        # Add topology-specific parameters
        kwargs_dict.update(self.topology_specific_kwargs)
        
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
        """Save the current state of the loss function."""
        state = {}
        state['critic_state_dict'] = self.critic.state_dict() if self.critic is not None else None
        state['critic_optimizer_state_dict'] = self.critic_optimizer.state_dict() if self.critic_optimizer is not None else None
        state['base_loss_state_dict'] = self.base_loss_state_dict
        
        # Save scheduler states
        state['scheduler_states'] = {}
        for param_name, scheduler in self.schedulers.items():
            state['scheduler_states'][param_name] = scheduler.state_dict()
                
        return state

    def load_state_dict(self, state_dict):
        """Load a previously saved state."""
        if self.critic is not None and state_dict.get('critic_state_dict') is not None:
            self.critic.load_state_dict(state_dict['critic_state_dict'])
        if self.critic_optimizer is not None and state_dict.get('critic_optimizer_state_dict') is not None:
            self.critic_optimizer.load_state_dict(state_dict['critic_optimizer_state_dict'])
        
        if 'base_loss_state_dict' in state_dict:
            self.base_loss_state_dict = state_dict['base_loss_state_dict']
            self.base_loss_f.load_state_dict(self.base_loss_state_dict)
        
        # Load scheduler states
        if 'scheduler_states' in state_dict:
            for param_name, scheduler_state in state_dict['scheduler_states'].items():
                if param_name in self.schedulers:
                    self.schedulers[param_name].load_state_dict(scheduler_state)

    @abstractmethod
    def _generate_group_action_parameters(self, data_num, latent_dim, selected_component_indices, **kwargs):
        """Generate topology-specific group action parameters."""
        pass

    @abstractmethod
    def _apply_group_action_latent_space(self, action_params, latent_rep, **kwargs):
        """Apply topology-specific group action in latent space."""
        pass

    def _select_latent_components(self, component_order, kl_components, prob_threshold=None):
        """
        Select latent components for group action.
        
        Default implementation uses the shared base_utils function.
        Subclasses can override if topology-specific selection is needed.
        """
        return select_latent_components(component_order, kl_components, prob_threshold)

    @abstractmethod
    def _get_critic_class(self):
        """Return the appropriate Critic class for this topology."""
        pass

    def _group_action_commutative_loss(self, data, model, kl_components):
        """
        Compute commutative group action loss: g.g'.x = g'.g.x
        
        This method implements the common structure while delegating topology-specific
        operations to abstract methods.
        """
        z = model.get_representations(data, is_deterministic=self.deterministic_rep)
        _, latent_dim = z.shape

        # Select components for group actions
        selected_row_indices, selected_component_indices = None, None
        for comp_order in reversed(range(2, self.commutative_component_order + 1)):
            selected_row_indices, selected_component_indices = self._select_latent_components(
                component_order=comp_order,
                kl_components=kl_components,
                prob_threshold=self.comp_latent_select_threshold
            )
            if selected_row_indices is not None and selected_component_indices is not None:
                break

        if selected_row_indices is None or selected_component_indices is None:
            raise ValueError(
                "No components selected based on the provided KL components and threshold in _group_action_commutative_loss"
            )

        if len(selected_row_indices) != len(z):
            z = z[selected_row_indices]

        # Split components for g and g'
        g_component_index = selected_component_indices[:, 0].unsqueeze(1)
        g_prime_component_indices = selected_component_indices[:, 1:]

        # Generate transformation parameters
        g_params = self._generate_group_action_parameters(
            data_num=len(selected_row_indices),
            latent_dim=latent_dim,
            selected_component_indices=g_component_index
        )
        
        gprime_params = self._generate_group_action_parameters(
            data_num=len(selected_row_indices),
            latent_dim=latent_dim,
            selected_component_indices=g_prime_component_indices
        )

        # Compute g.g'.x
        z_g = self._apply_group_action_latent_space(g_params, z)
        x_g = model.reconstruct_latents(z_g)
        z_g_encoded = model.get_representations(x_g, is_deterministic=self.deterministic_rep)
        z_ggprime = self._apply_group_action_latent_space(gprime_params, z_g_encoded)
        x_ggprime = model.reconstruct_latents(z_ggprime)

        # Compute g'.g.x
        z_gprime = self._apply_group_action_latent_space(gprime_params, z)
        x_gprime = model.reconstruct_latents(z_gprime)
        z_gprime_encoded = model.get_representations(x_gprime, is_deterministic=self.deterministic_rep)
        z_gprimeg = self._apply_group_action_latent_space(g_params, z_gprime_encoded)
        x_gprimeg = model.reconstruct_latents(z_gprimeg)

        # Compute comparison loss
        if self.commutative_comparison_dist == 'gaussian':
            commutative_loss = F.mse_loss(x_ggprime, x_gprimeg, reduction='sum') / data.size(0)
        elif self.commutative_comparison_dist == 'bernoulli':
            # TODO: Implement bernoulli loss
            raise NotImplementedError("Bernoulli comparison loss not yet implemented")
        else:
            raise ValueError(f"Unknown comparison distribution: {self.commutative_comparison_dist}")

        return commutative_loss

    def _apply_multiple_group_actions_images(self, real_images, model):
        """
        Apply multiple consecutive group actions on images through latent space.
        """
        fake_images = real_images.clone()

        for _ in range(self.meaningful_transformation_order):
            # Encode current images
            z = model.get_representations(fake_images, is_deterministic=self.deterministic_rep)
            
            # Calculate KL components for current images
            current_kl_components = self._compute_kl_components(fake_images, model)

            latent_dim = z.size(1)

            # Select components for group action
            selected_row_indices, selected_component_indices = None, None
            for comp_order in reversed(range(1, self.meaningful_component_order + 1)):
                selected_row_indices, selected_component_indices = self._select_latent_components(
                    component_order=comp_order,
                    kl_components=current_kl_components,
                    prob_threshold=self.comp_latent_select_threshold
                )
                if selected_row_indices is not None and selected_component_indices is not None:
                    break
        
            if selected_row_indices is None or selected_component_indices is None:
                raise ValueError(
                    "No components selected based on the provided KL components and threshold in group_action_meaningful_loss"
                )
        
            if len(selected_row_indices) != len(z):
                z = z[selected_row_indices]

            # Generate and apply group action
            group_params = self._generate_group_action_parameters(
                data_num=len(selected_row_indices),
                latent_dim=latent_dim,
                selected_component_indices=selected_component_indices
            )
        
            z_transformed = self._apply_group_action_latent_space(group_params, z)
            fake_images = model.reconstruct_latents(z_transformed)

        return fake_images

    def _initialize_critic_if_needed(self, data):
        """Initialize critic and optimizer if not already done."""
        if self.critic is None:
            input_channels_num = data.shape[1]
            CriticClass = self._get_critic_class()
            self.critic = CriticClass(input_channels_num=input_channels_num).to(self.device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.meaningful_critic_lr)

    def _train_critic(self, data, model, log_data):
        """Train the critic for meaningful loss."""
        d_losses = torch.zeros(self.meaningful_n_critic, device=data.device)
        
        for i in range(self.meaningful_n_critic):
            # Generate fake images without gradient for critic training
            with torch.no_grad():
                fake_images = self._apply_multiple_group_actions_images(data, model)

            critic_real = self.critic(data)
            critic_fake = self.critic(fake_images)
            gp = self.critic._compute_gradient_penalty(data, fake_images)
            
            # WGAN-GP critic loss
            d_loss = -(critic_real.mean() - critic_fake.mean()) * self.meaningful_weight + \
                     self.meaningful_critic_gradient_penalty_weight * gp

            self.critic_optimizer.zero_grad()
            d_loss.backward()
            self.critic_optimizer.step()
            d_losses[i] = d_loss.item()

        log_data['critic_loss'] = d_losses.mean().item()

    def _compute_generator_loss(self, data, model, log_data):
        """Compute generator loss for meaningful training."""
        # Generate fake images with gradients for generator update
        fake_images = self._apply_multiple_group_actions_images(data, model)

        # Freeze critic for generator update
        for p in self.critic.parameters():
            p.requires_grad_(False)

        # WGAN generator loss
        critic_fake = self.critic(fake_images)
        g_loss = -critic_fake.mean()
        log_data['generator_loss'] = g_loss.item()

        # Unfreeze critic
        for p in self.critic.parameters():
            p.requires_grad_(True)

        return g_loss

    def _compute_kl_components(self, data, model):
        """
        Compute KL components compatible with both n_vae and s_n_vae models.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data batch
        model : nn.Module
            The VAE model (n_vae or s_n_vae)
            
        Returns
        -------
        torch.Tensor
            KL components for each latent factor
        """
        with torch.no_grad():
            stats_qzx = model.encoder(data)['stats_qzx']
            
        # Use topology information from the model
        # S_N_VAE case: parse mixed topology parameters
        latent_factor_topologies = self.latent_factor_topologies
        
        # Calculate parameter counts for each factor type
        factor_params = []
        for topology in latent_factor_topologies:
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
                f"for topologies {latent_factor_topologies}."
            )
        
        # Calculate KL components for each factor
        kl_components_list = []
        start_idx = 0
        
        for topology, n_params in zip(latent_factor_topologies, factor_params):
            end_idx = start_idx + n_params
            factor_params_tensor = stats_qzx[:, start_idx:end_idx]
            
            if topology == 'R1':
                # For Normal distribution, params are (mean, logvar)
                mean = factor_params_tensor[:, 0]
                logvar = factor_params_tensor[:, 1]
                kl_component = kl_normal_loss(mean, logvar, raw=True)
            elif topology == 'S1':
                # For Power Spherical distribution, params are (mu_x, mu_y, kappa)
                # Use reduction='none' to get per-sample KL values
                kl_component = kl_power_spherical_uniform_factor_wise(factor_params_tensor, reduction='none')
            else:
                raise ValueError(f"Unknown latent factor topology: {topology}")
            
            kl_components_list.append(kl_component)
            start_idx = end_idx
        
        # Stack to create tensor with shape [batch_size, num_factors]
        kl_components_raw = torch.cat(kl_components_list, dim=1)

        print(f"KL components shape: {kl_components_raw.shape}")
        
        return kl_components_raw

    def __call__(self, data, model, vae_optimizer):
        """Main training loop with common structure."""
        # Initialize topology information from model
        self.latent_factor_topologies = model.latent_factor_topologies

        log_data = OrderedDict()
        base_loss = 0

        # Handle base loss computation
        if self.base_loss_f.mode == 'post_forward':
            model_out = model(data)
            inputs = {
                'data': data,
                **model_out,
                'is_train': True,
            }
            loss_out = self.base_loss_f(**inputs)
            base_loss = loss_out['loss']
            log_data.update(loss_out['to_log'])

        elif self.base_loss_f.mode == 'pre_forward':
            inputs = {
                'model': model,
                'data': data,
                'is_train': True,
            }
            loss_out = self.base_loss_f(**inputs)
            base_loss = loss_out['loss']
            log_data.update(loss_out['to_log'])

        elif self.base_loss_f.mode == 'optimizes_internally':
            raise NotImplementedError("This loss function is not compatible with 'optimizes_internally' mode for baseloss")

        # Get KL components
        kl_components_raw = self._compute_kl_components(data, model)

        # Increment step counter
        self.current_step += 1
        
        # Group action losses (only after warm-up)
        group_loss = 0
        in_warm_up = self.current_step <= self.warm_up_steps

        # Commutative loss
        if self.commutative_weight > 0 and not in_warm_up:
            g_commutative_loss = self._group_action_commutative_loss(data, model, kl_components_raw)
            log_data['g_commutative_loss'] = g_commutative_loss.item()
            group_loss += self.commutative_weight * g_commutative_loss

        # Meaningful loss
        if self.meaningful_weight > 0 and not in_warm_up:
            self._initialize_critic_if_needed(data)
            
            # Train critic
            self._train_critic(data, model, log_data)
            
            # Compute generator loss
            g_loss = self._compute_generator_loss(data, model, log_data)
            group_loss += self.meaningful_weight * g_loss
            
            total_loss = base_loss + group_loss
            vae_optimizer.zero_grad()
            total_loss.backward()
            vae_optimizer.step()

            log_data['loss'] = total_loss.item()
            return {'loss': total_loss, 'to_log': log_data}
        
        # Standard optimization without meaningful loss
        total_loss = base_loss + group_loss
        vae_optimizer.zero_grad()
        total_loss.backward()
        vae_optimizer.step()

        log_data['loss'] = total_loss.item()
        return {'loss': total_loss, 'to_log': log_data}
