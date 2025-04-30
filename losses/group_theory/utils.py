import torch
import torch.nn as nn
import torch.nn.functional as F

def select_latent_components(component_order, kl_components):
    """
    Selects a subset of latent components based on the provided component order and KL components.

    Args:
        component_order (int): The number of latent dimensions to randomly select.
        kl_components (torch.Tensor): KL divergence values for each component. Shape (batch, latent_dim).

    Returns:
        torch.Tensor: A tensor of shape (batch_size, component_order) containing the selected indices.
    """
    # Ensure inputs are on the correct device (assuming kl_components determines the device)
    device = kl_components.device

    # Calculate selection probabilities using softmax
    probs = F.softmax(kl_components, dim=1)

    # Sample 'component_order' indices for each batch item based on probabilities
    selected_components = torch.multinomial(probs, component_order, replacement=False)

    return selected_components


def generate_latent_translations_selected_components(batch_size, latent_dim, selected_components_indices, selected_components_variances):
    """
    Generates random translation parameters for latent space transformation.

    In this function, we randomly select a subset of dimensions in the latent space based on the
    `selected_components_indices` and modify them by sampling from a Gaussian distribution with mean 0 and
    variance specified by `selected_components_variances`.

    Args:
        batch_size (int): The number of transformation vectors to generate.
        latent_dim (int): The total dimensionality of the latent space.
        selected_components_indices (torch.Tensor): Indices of the selected components. Shape (batch, component_order).
        selected_components_variances (torch.Tensor): Variances for each selected component. Shape (batch, component_order).

    Returns:
        torch.Tensor: A tensor of shape (batch_size, latent_dim) containing the
                      random translation parameters. Only `component_order`
                      dimensions per vector will have non-zero values drawn from N(0, selected_components_variances).
    """
    # Ensure inputs are on the correct device (assuming kl_components determines the device)
    device = selected_components_indices.device

    # Initialize transformation parameters with zeros
    transformation_parameters = torch.zeros(batch_size, latent_dim, device=device)

    # Get the variances for the selected components
    selected_variances = selected_components_variances

    # Sample from N(0, 1)
    random_samples = torch.randn(batch_size, selected_components_indices.size(1), device=device)

    # Scale samples by the standard deviation (sqrt of variance)
    transformation_values = random_samples * torch.sqrt(selected_variances)

    # Place the sampled and scaled values into the transformation_parameters tensor
    transformation_parameters.scatter_(1, selected_components_indices, transformation_values)

    return transformation_parameters


def generate_latent_translations(batch_size, latent_dim, component_order, kl_components, variance_components):
    """
    Generates latent translations by combining component selection and parameter generation.

    This higher-level function coordinates:
    1. Component selection using KL-based probabilities
    2. Variance gathering for selected components
    3. Translation parameter generation using selected components

    Args:
        batch_size (int): Number of transformation vectors to generate
        latent_dim (int): Total dimensionality of latent space
        component_order (int): Number of components to select/modify per sample
        kl_components (torch.Tensor): Component selection weights (batch, latent_dim)
        variance_components (torch.Tensor): Per-component variance values (batch, latent_dim)

    Returns:
        torch.Tensor: Translation parameters tensor of shape (batch_size, latent_dim)
                      with non-zero values only in selected components, sampled from
                      N(0, variance_components[selected_indices])
    """
    # Ensure inputs are on the correct device (assuming kl_components determines the device)
    device = kl_components.device
    variance_components = variance_components.to(device)

    # Select components and generate translations using helper functions
    selected_indices = select_latent_components(component_order, kl_components)
    selected_variances = variance_components.gather(1, selected_indices)
    
    return generate_latent_translations_selected_components(
        batch_size, 
        latent_dim,
        selected_indices,
        selected_variances
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

