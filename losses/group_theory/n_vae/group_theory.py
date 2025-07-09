import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_group_theory import BaseGroupTheoryLoss
from ..base_utils import Critic
from .utils import (generate_latent_translations_selected_components,
                    apply_group_action_latent_space
                    )


class GroupTheoryNVAELoss(BaseGroupTheoryLoss):
    """    
    Group theory losses for R^1 topology (translation-only) VAE models.
    
    This implementation is specialized for models with R^1 latent factors that only
    support translation operations in latent space.
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
                 group_action_latent_range=2,
                 group_action_latent_distribution='uniform',
                 comp_latent_select_threshold=0,
                 base_loss_state_dict=None,
                 warm_up_steps=0,
                 schedulers_kwargs=None,
                 **kwargs):
        
        # Store R^1-specific parameters
        self.group_action_latent_range = group_action_latent_range
        self.group_action_latent_distribution = group_action_latent_distribution
        
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
    def topology_specific_kwargs(self):
        """Return R^1-specific parameters for serialization."""
        return {
            'group_action_latent_range': self.group_action_latent_range,
            'group_action_latent_distribution': self.group_action_latent_distribution,
        }

    @property
    def name(self):
        return "group_theory_n_vae"

    def _get_critic_class(self):
        """Return the Critic class for R^1 topology."""
        return Critic

    def _generate_group_action_parameters(self, data_num, latent_dim, selected_component_indices, **kwargs):
        """Generate R^1 translation parameters."""
        return generate_latent_translations_selected_components(
            data_num=data_num,
            latent_dim=latent_dim,
            selected_components_indices=selected_component_indices,
            range=self.group_action_latent_range,
            distribution=self.group_action_latent_distribution,
        )

    def _apply_group_action_latent_space(self, action_params, latent_rep, **kwargs):
        """Apply R^1 group action (simple addition)."""
        return apply_group_action_latent_space(action_params, latent_rep)

