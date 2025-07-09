import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_group_theory import BaseGroupTheoryLoss
from ..base_utils import Critic
from .utils import (apply_group_action_latent_space,
                    generate_group_action_parameters)


class GroupTheorySNVAELoss(BaseGroupTheoryLoss):
    """    
    Group theory losses for mixed R^1 and S^1 topology VAE models.
    
    This implementation supports models with mixed latent factors that can have
    both R^1 (translation) and S^1 (rotation) topologies.
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
                 g_action_r1_range=2,
                 g_action_s1_range=2 * torch.pi,
                 g_action_r1_dist='uniform',
                 g_action_s1_dist='uniform',
                 comp_latent_select_threshold=0,
                 base_loss_state_dict=None,
                 warm_up_steps=0,
                 schedulers_kwargs=None,
                 **kwargs):
        
        # Store mixed topology parameters
        self.g_action_r1_range = g_action_r1_range
        self.g_action_s1_range = g_action_s1_range
        self.g_action_r1_dist = g_action_r1_dist
        self.g_action_s1_dist = g_action_s1_dist
        
        # Initialize base class
        super().__init__(
            base_loss_name=base_loss_name,
            base_loss_kwargs=base_loss_kwargs,
            rec_dist=rec_dist,
            device=device,
            commutative_weight=commutative_weight,
            commutative_component_order=commutative_component_order,
            commutative_comparison_dist=commutative_comparison_dist,
            meaningful_weight=meaningful_weight,
            meaningful_component_order=meaningful_component_order,
            meaningful_transformation_order=meaningful_transformation_order,
            meaningful_critic_gradient_penalty_weight=meaningful_critic_gradient_penalty_weight,
            meaningful_critic_lr=meaningful_critic_lr,
            meaningful_n_critic=meaningful_n_critic,
            deterministic_rep=deterministic_rep,
            comp_latent_select_threshold=comp_latent_select_threshold,
            base_loss_state_dict=base_loss_state_dict,
            warm_up_steps=warm_up_steps,
            schedulers_kwargs=schedulers_kwargs,
            **kwargs
        )

    @property
    def name(self):
        return "group_theory_s_n_vae"

    @property
    def topology_specific_kwargs(self):
        """Return mixed topology-specific parameters for serialization."""
        return {
            'g_action_r1_range': self.g_action_r1_range,
            'g_action_s1_range': self.g_action_s1_range,
            'g_action_r1_dist': self.g_action_r1_dist,
            'g_action_s1_dist': self.g_action_s1_dist,
        }

    def _get_critic_class(self):
        """Return the Critic class for mixed topology."""
        return Critic

    def _generate_group_action_parameters(self, data_num, latent_dim, selected_component_indices, **kwargs):
        """Generate mixed topology group action parameters."""
        if self.latent_factors_topologies is None:
            raise ValueError("latent_factors_topologies must be set before generating group action parameters")
            
        return generate_group_action_parameters(
            data_num=data_num,
            latent_dim=latent_dim,
            selected_component_indices=selected_component_indices,
            latent_factor_topologies=self.latent_factors_topologies,
            r1_range=self.g_action_r1_range,
            s1_range=self.g_action_s1_range,
            r1_dist=self.g_action_r1_dist,
            s1_dist=self.g_action_s1_dist,
        )

    def _apply_group_action_latent_space(self, action_params, latent_rep, **kwargs):
        """Apply mixed topology group action."""
        if self.latent_factors_topologies is None:
            raise ValueError("latent_factors_topologies must be set before applying group actions")
            
        return apply_group_action_latent_space(action_params, latent_rep, self.latent_factors_topologies)

