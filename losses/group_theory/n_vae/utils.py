import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_utils import select_latent_components

def generate_latent_translations_selected_components(data_num,
                                                      factor_num, 
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
        factor_num (int): The total number of latent factors (latent space dimensionality).
        selected_components_indices (torch.Tensor): Indices of the selected components. Shape (batch, component_order).
        range (float): For uniform distribution: the range [-range, range] from which to sample translation values.
                      For normal distribution: the standard deviation of the normal distribution.
        distribution (str): The distribution to sample from. Either 'uniform' or 'normal'. Default is 'uniform'.

    Returns:
        torch.Tensor: A tensor of shape (data_num, factor_num) containing the
                      random translation parameters. Only `component_order`
                      dimensions per vector will have non-zero values drawn from the specified distribution.
    """
    # Ensure inputs are on the correct device (assuming kl_components determines the device)
    device = selected_components_indices.device

    # Initialize transformation parameters with zeros
    transformation_parameters = torch.zeros(data_num, factor_num, device=device)

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

def apply_group_action_latent_space(transformation_parameters, latent_space):
    """
    Apply group action on latent space using transformation parameters.
    """
    # Ensure parameters are on the same device as the latent space
    transformation_parameters = transformation_parameters.to(latent_space.device)
    transformed_latent_space = latent_space + transformation_parameters
    return transformed_latent_space
