import torch
import torch.nn as nn
import torch.nn.functional as F

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


def generate_latent_translations_selected_components(data_num,
                                                      latent_dim, 
                                                      selected_components_indices, 
                                                      range=1, 
                                                      distribution='normal'
                                                      ):
    """
    Generates random translation parameters for latent space transformation.

    In this function, we randomly select a subset of dimensions in the latent space based on the
    `selected_components_indices` and modify them by sampling from either a uniform or normal distribution.

    Args:
        data_num (int): The number of transformation vectors to generate.
        latent_dim (int): The total dimensionality of the latent space.
        selected_components_indices (torch.Tensor): Indices of the selected components. Shape (batch, component_order).
        range (float): For uniform distribution: the range [-range, range] from which to sample translation values.
                      For normal distribution: the standard deviation of the normal distribution.
        distribution (str): The distribution to sample from. Either 'uniform' or 'normal'. Default is 'uniform'.

    Returns:
        torch.Tensor: A tensor of shape (data_num, latent_dim) containing the
                      random translation parameters. Only `component_order`
                      dimensions per vector will have non-zero values drawn from the specified distribution.
    """
    # Ensure inputs are on the correct device (assuming kl_components determines the device)
    device = selected_components_indices.device

    # Initialize transformation parameters with zeros
    transformation_parameters = torch.zeros(data_num, latent_dim, device=device)

    # Sample from the specified distribution
    if distribution == 'uniform':
        # Sample from uniform distribution in [-range, range]
        random_samples = (2 * torch.rand(data_num, selected_components_indices.size(1), device=device) - 1) * range
    elif distribution == 'normal':
        # Sample from normal distribution with mean=0 and std=range
        random_samples = torch.randn(data_num, selected_components_indices.size(1), device=device) * range
    else:
        raise ValueError(f"Unsupported distribution '{distribution}'. Must be 'uniform' or 'normal'.")

    transformation_values = random_samples

    # Place the sampled values into the transformation_parameters tensor
    transformation_parameters.scatter_(1, selected_components_indices, transformation_values)

    return transformation_parameters


def generate_latent_translations(data_num, 
                                 latent_dim, 
                                 component_order, 
                                 kl_components, 
                                 range=1, 
                                 distribution='normal'
                                 ):
    """
    Generates latent translations by combining component selection and parameter generation.

    This higher-level function coordinates:
    1. Component selection using KL-based probabilities
    2. Variance gathering for selected components
    3. Translation parameter generation using selected components

    Args:
        data_num (int): Number of transformation vectors to generate
        latent_dim (int): Total dimensionality of latent space
        component_order (int): Number of components to select/modify per sample
        kl_components (torch.Tensor): Component selection weights (batch, latent_dim)
        range (float): For uniform distribution: the range [-range, range] from which to sample translation values.
                      For normal distribution: the standard deviation of the normal distribution.
        distribution (str): The distribution to sample from. Either 'uniform' or 'normal'. Default is 'uniform'.

    Returns:
        torch.Tensor: Translation parameters tensor of shape (data_num, latent_dim)
                      with non-zero values only in selected components, sampled from
                      the specified distribution.
    """
    # Ensure inputs are on the correct device (assuming kl_components determines the device)
    device = kl_components.device

    # Select components and generate translations using helper functions
    selected_indices = select_latent_components(component_order, kl_components)
    
    return generate_latent_translations_selected_components(
        data_num, 
        latent_dim,
        selected_indices,
        range=range,
        distribution=distribution
    )


def apply_group_action_latent_space(transformation_parameters, latent_space):
    """
    Apply group action on latent space using transformation parameters.
    """
    # Ensure parameters are on the same device as the latent space
    transformation_parameters = transformation_parameters.to(latent_space.device)
    transformed_latent_space = latent_space + transformation_parameters
    return transformed_latent_space

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

