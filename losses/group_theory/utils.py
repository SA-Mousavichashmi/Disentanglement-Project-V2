import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def select_latent_components(component_order: int,
                             kl_components: torch.Tensor,
                             prob_threshold: float | None = None):
    device = kl_components.device
    probs = F.softmax(kl_components, dim=1)          # 1️⃣

    if prob_threshold is not None:
        # keep probabilities ≥ threshold, zero-out the rest
        probs = torch.where(probs >= prob_threshold, probs, torch.zeros_like(probs))
        row_mask = (probs.sum(1) > 0)                # at least one surviving dim
        probs = probs[row_mask]                      # 2️⃣ keep only valid rows
        if probs.size(0) == 0:
            return None, None
        probs = probs / probs.sum(1, keepdim=True)   # 3️⃣ renormalize WITHOUT soft-max
    else:
        row_mask = torch.ones(kl_components.size(0), dtype=torch.bool, device=device)

    sel_rows = torch.where(row_mask)[0]
    if probs.size(1) < component_order:
        return None, None

    sel_components = torch.multinomial(probs, component_order, replacement=False)
    return sel_rows, sel_components


def generate_group_action_parameters(data_num,
                                    latent_dim,
                                    selected_components_indices,
                                    latent_factor_topologies,
                                    r1_range=1,
                                    s1_range= 2 * torch.pi,
                                    r1_distribution='normal',
                                    s1_distribution='uniform'
                                    ):
    """
    Generates random group action parameters for latent space transformation.
    Supports both R^1 (translation) and S^1 (rotation) topologies.

    Args:
        data_num (int): The number of transformation vectors to generate.
        latent_dim (int): The total dimensionality of the latent space.
        selected_components_indices (torch.Tensor): Indices of the selected components. Shape (batch, component_order).
        latent_factor_topologies (list): List of topology types for each latent factor. 
                                       Each element is either 'R^1' or 'S^1'. If None, defaults to all 'R^1'.
        r1_range (float): For uniform distribution: the range [-r1_range, r1_range] for R^1.
                         For normal distribution: the standard deviation for R^1.
        s1_range (float): For uniform distribution: the range [0, s1_range] for S^1 angles.
                         For normal distribution: the standard deviation for S^1.
        r1_distribution (str): The distribution to sample R^1 parameters from. Either 'uniform' or 'normal'.
        s1_distribution (str): The distribution to sample S^1 parameters from. Either 'uniform' or 'normal'.

    Returns:
        dict: A dictionary containing:
            - 'r1_parameters': torch.Tensor of shape (data_num, latent_dim) for R^1 translation parameters
            - 's1_parameters': torch.Tensor of shape (data_num, num_s1_factors) for S^1 rotation angles
            - 's1_factor_indices': list of indices corresponding to S^1 factors
    """
    device = selected_components_indices.device
    
    # Initialize parameters
    r1_parameters = torch.zeros(data_num, latent_dim, device=device)
    s1_factor_indices = [i for i, topology in enumerate(latent_factor_topologies) if topology == 'S^1']
    s1_parameters = torch.zeros(data_num, len(s1_factor_indices), device=device)
    
    # Generate random samples for R^1
    if r1_distribution == 'uniform':
        r1_samples = (2 * torch.rand(data_num, selected_components_indices.size(1), device=device) - 1) * r1_range
    elif r1_distribution == 'normal':
        r1_samples = torch.randn(data_num, selected_components_indices.size(1), device=device) * r1_range
    else:
        raise ValueError(f"Unsupported R^1 distribution '{r1_distribution}'. Must be 'uniform' or 'normal'.")

    # Generate random samples for S^1
    if s1_distribution == 'uniform':
        s1_samples = torch.rand(data_num, selected_components_indices.size(1), device=device) * s1_range
    elif s1_distribution == 'normal':
        s1_samples = torch.randn(data_num, selected_components_indices.size(1), device=device) * s1_range
    else:
        raise ValueError(f"Unsupported S^1 distribution '{s1_distribution}'. Must be 'uniform' or 'normal'.")
    
    # Assign parameters based on topology of selected components
    for batch_idx in range(data_num):
        for comp_idx, selected_comp in enumerate(selected_components_indices[batch_idx]):
            selected_comp = selected_comp.item()
            topology = latent_factor_topologies[selected_comp] if selected_comp < len(latent_factor_topologies) else 'R^1'
            
            if topology == 'R^1':
                r1_parameters[batch_idx, selected_comp] = r1_samples[batch_idx, comp_idx]
            elif topology == 'S^1':
                if selected_comp in s1_factor_indices:
                    s1_idx = s1_factor_indices.index(selected_comp)
                    s1_parameters[batch_idx, s1_idx] = s1_samples[batch_idx, comp_idx]
    
    return {
        'r1_parameters': r1_parameters,
        's1_parameters': s1_parameters,
        's1_factor_indices': s1_factor_indices
    }


def apply_group_action_latent_space(group_action_params, latent_space, latent_factor_topologies=None):
    """
    Apply topology-aware group action on latent space.
    Supports both R^1 (translation) and S^1 (rotation) topologies.

    Args:
        group_action_params (dict): Dictionary containing group action parameters with keys:
            - 'r1_parameters': torch.Tensor of shape (batch_size, latent_dim) for R^1 translations
            - 's1_parameters': torch.Tensor of shape (batch_size, num_s1_factors) for S^1 rotation angles
            - 's1_factor_indices': list of indices corresponding to S^1 factors
        latent_space (torch.Tensor): Input latent representations.
            For mixed topology: shape (batch_size, total_latent_dim)
            For S^1 only: shape (batch_size, latent_factor_num * 2) where each factor uses 2D (cos, sin)
        latent_factor_topologies (list): List of topology types for each latent factor.

    Returns:
        torch.Tensor: Transformed latent space with same shape as input.
    """
    device = latent_space.device
    batch_size = latent_space.size(0)
    
    # Default to all R^1 if no topology information provided
    if latent_factor_topologies is None:
        latent_factor_topologies = ['R^1'] * latent_space.size(1)
    
    # Extract parameters
    r1_parameters = group_action_params['r1_parameters'].to(device)
    s1_parameters = group_action_params['s1_parameters'].to(device)
    s1_factor_indices = group_action_params['s1_factor_indices']
    
    transformed_latent = latent_space.clone()
    
    # Apply R^1 translations (addition)
    transformed_latent = transformed_latent + r1_parameters
    
    # Apply S^1 rotations if there are any S^1 factors
    if len(s1_factor_indices) > 0 and s1_parameters.size(1) > 0:
        _apply_s1_rotations(transformed_latent, s1_parameters, s1_factor_indices, latent_factor_topologies)
    
    return transformed_latent


def _apply_s1_rotations(latent_tensor, s1_parameters, s1_factor_indices, latent_factor_topologies):
    """
    Helper function to apply S^1 rotations to latent tensor in-place.
    Supports both toroidal VAE format and mixed topology format.
    
    Args:
        latent_tensor (torch.Tensor): Latent tensor to modify in-place
        s1_parameters (torch.Tensor): S^1 rotation angles
        s1_factor_indices (list): Indices of S^1 factors
        latent_factor_topologies (list): Topology information
    """
    batch_size = latent_tensor.size(0)
    num_s1_factors = len(s1_factor_indices)
    
    # Check if latent space is in flattened format (for toroidal VAE)
    if latent_tensor.size(1) == num_s1_factors * 2:
        # Toroidal VAE format: (batch_size, latent_factor_num * 2)
        latent_reshaped = latent_tensor.view(batch_size, num_s1_factors, 2)
        
        for i, factor_idx in enumerate(s1_factor_indices):
            if i < s1_parameters.size(1):
                rotation_angle = s1_parameters[:, i]
                _rotate_s1_factor_2d(latent_reshaped[:, i, :], rotation_angle)
        
        # Update original tensor (in-place)
        latent_tensor.copy_(latent_reshaped.view(batch_size, -1))
    else:
        # Mixed topology format or other formats
        for i, factor_idx in enumerate(s1_factor_indices):
            if i < s1_parameters.size(1) and factor_idx * 2 + 1 < latent_tensor.size(1):
                rotation_angle = s1_parameters[:, i]
                cos_sin_pair = latent_tensor[:, factor_idx * 2:factor_idx * 2 + 2]
                _rotate_s1_factor_2d(cos_sin_pair, rotation_angle)


def _rotate_s1_factor_2d(cos_sin_tensor, rotation_angle):
    """
    Helper function to rotate a 2D (cos, sin) representation on S^1 in-place.
    
    Args:
        cos_sin_tensor (torch.Tensor): Tensor of shape (..., 2) containing (cos, sin) values
        rotation_angle (torch.Tensor): Rotation angles to apply
    """
    cos_val = cos_sin_tensor[..., 0]
    sin_val = cos_sin_tensor[..., 1]
    
    # Convert to angle, apply rotation, convert back
    current_angle = torch.atan2(sin_val, cos_val)
    new_angle = current_angle + rotation_angle
    
    # Update with rotated values (in-place)
    cos_sin_tensor[..., 0] = torch.cos(new_angle)
    cos_sin_tensor[..., 1] = torch.sin(new_angle)


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

