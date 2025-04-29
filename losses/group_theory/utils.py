import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_random_latent_translation(batch_size, latent_dim, component_order, kl_components, variance_components):
    """
    Generates random translation parameters for latent space transformation.

    In this function, we randomly select a subset of dimensions in the latent space based on the
    `component_order` and the probability based (softmax) on the kl_components.
    The selected dimensions are then modified by sampling from a Gaussian distribution with mean 0 and
    variance specified by `variance_components`.

    Args:
        batch_size (int): The number of transformation vectors to generate.
        latent_dim (int): The total dimensionality of the latent space.
        component_order (int): The number of latent dimensions to randomly select
                               and modify (translate) based on the component order and probability derived (softmax) from kl_components.
        kl_components (torch.Tensor): KL divergence values for each component based on that the probability for selecting components is chosen using softmax. Shape (batch, latent_dim).
        variance_components (torch.Tensor): The variances for each components that is sampled from gaussian distribution with (mean 0, variance) for each component. Shape (batch, latent_dim).

    Returns:
        torch.Tensor: A tensor of shape (batch_size, latent_dim) containing the
                      random translation parameters. Only `component_order`
                      dimensions per vector will have non-zero values drawn from N(0, variance_components[selected_indices]).
    """
    # Ensure inputs are on the correct device (assuming kl_components determines the device)
    device = kl_components.device
    variance_components = variance_components.to(device)

    # Calculate selection probabilities using softmax
    probs = F.softmax(kl_components, dim=0)

    # Sample 'component_order' indices for each batch item based on probabilities
    # probs.repeat(batch_size, 1) creates a (batch_size, latent_dim) tensor of probabilities
    # replacement=False ensures unique indices are selected for each batch item if component_order < latent_dim
    selected_indices = torch.multinomial(probs.repeat(batch_size, 1), component_order, replacement=False)
    # selected_indices shape: (batch_size, component_order)

    # Initialize transformation parameters with zeros
    transformation_parameters = torch.zeros(batch_size, latent_dim, device=device)

    # Get the variances for the selected components
    selected_variances = variance_components[selected_indices]
    # selected_variances shape: (batch_size, component_order)

    # Sample from N(0, 1)
    random_samples = torch.randn(batch_size, component_order, device=device)

    # Scale samples by the standard deviation (sqrt of variance)
    transformation_values = random_samples * torch.sqrt(selected_variances) # TODO, this must be checked for getting square root of variance

    # Place the sampled and scaled values into the transformation_parameters tensor
    # scatter_(dim, index, src) -> self[index[i][j]][j] = src[i][j] for dim=0
    # scatter_(dim, index, src) -> self[i][index[i][j]] = src[i][j] for dim=1
    transformation_parameters.scatter_(1, selected_indices, transformation_values)

    return transformation_parameters


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
                nn.ReLU(),
                nn.Conv2d(32, 32, 4, stride=2, padding=1),  # 32x32x32 -> 16x16x32
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16x16x32 -> 8x8x64
                nn.ReLU(),
                nn.Conv2d(64, 64, 4, stride=2, padding=1),  # 8x8x64 -> 4x4x64
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(4 * 4 * 64, 256),
                nn.ReLU(),
                nn.Linear(256, 1)  # Output scalar value
            )
        
        elif architecture == 'burgess':
            self.critic = nn.Sequential(
                nn.Conv2d(input_channels_num, 32, 4, stride=2, padding=1),  # 64x64xC -> 32x32x32
                nn.ReLU(),
                nn.Conv2d(32, 32, 4, stride=2, padding=1),                  # 32x32x32 -> 16x16x32
                nn.ReLU(),
                nn.Conv2d(32, 32, 4, stride=2, padding=1),                  # 16x16x32 -> 8x8x32
                nn.ReLU(),
                nn.Conv2d(32, 32, 4, stride=2, padding=1),                  # 8x8x32 -> 4x4x32
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(4 * 4 * 32, 256),
                nn.ReLU(),
                nn.Linear(256, 256),  # Output scalar value
                nn.ReLU(),
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

