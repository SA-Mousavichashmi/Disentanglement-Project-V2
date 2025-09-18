"""
Shared utilities for Group Theory losses to eliminate code duplication between n_vae and s_n_vae implementations.

This module contains common functions used by both R^1 and mixed topology group theory implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def select_latent_components(component_order: int,
                             kl_components: torch.Tensor,
                             prob_threshold: float | None = None):
    """
    Select latent components based on KL divergence probabilities.
    
    This function is shared between n_vae and s_n_vae group theory implementations
    to avoid code duplication.
    
    Args:
        component_order (int): Number of components to select per sample
        kl_components (torch.Tensor): KL divergence values per component (batch_size, latent_dim)
        prob_threshold (float, optional): Minimum probability threshold for component selection
        
    Returns:
        tuple: (selected_row_indices, selected_component_indices) or (None, None) if no valid selection
    """
    device = kl_components.device
    probs = kl_components / kl_components.sum(1, keepdim=True)  # normalize to get probabilities

    if prob_threshold is not None:
        # keep probabilities ≥ threshold, zero-out the rest
        probs = torch.where(probs >= prob_threshold, probs, torch.zeros_like(probs))
        row_mask = (probs.sum(1) > 0)                # at least one surviving dim
        probs = probs[row_mask]                      # keep only valid rows
        if probs.size(0) == 0:
            return None, None
        probs = probs / probs.sum(1, keepdim=True)   # renormalize WITHOUT soft-max
    else:
        row_mask = torch.ones(kl_components.size(0), dtype=torch.bool, device=device)

    sel_rows = torch.where(row_mask)[0]
    if probs.size(1) < component_order:
        return None, None

    sel_components = torch.multinomial(probs, component_order, replacement=False)
    return sel_rows, sel_components


################# Group Meaningful Loss Critic #################

class Critic(nn.Module):
    """
    Critic network for the WGAN-GP loss in the group meaningful loss.
    """
    def __init__(self, input_channels_num, architecture='locatello'):
        super(Critic, self).__init__()

        if architecture == 'locatello':
            self.critic = nn.Sequential(
            nn.Conv2d(input_channels_num, 32, 4, stride=2, padding=1),  # 64x64xC -> 32x32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # 32x32x32 -> 16x16x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16x16x32 -> 8x8x64
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),  # 8x8x64 -> 4x4x64
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 64, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # Output scalar value
            )

        elif architecture == 'burgess':
            self.critic = nn.Sequential(
            nn.Conv2d(input_channels_num, 32, 4, stride=2, padding=1),  # 64x64xC -> 32x32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),                  # 32x32x32 -> 16x16x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),                  # 16x16x32 -> 8x8x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),                  # 8x8x32 -> 4x4x32
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 32, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),  # Output scalar value
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # Output scalar value
            )

    def _compute_gradient_penalty(self, real_images, fake_images):
        """
        Compute the gradient penalty for WGAN-GP.
        """
        device = real_images.device
        batch_size = real_images.size(0)

        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        alpha = alpha.expand_as(real_images)

        interpolates = alpha * real_images + ((1 - alpha) * fake_images)
        interpolates.requires_grad_(True)

        disc_interpolates = self(interpolates)

        grad_outputs = torch.ones_like(disc_interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    def forward(self, x):
        return self.critic(x)
