import torch
from scipy import stats  # Add scipy import

####################### Latent Traversal Functions #########################

def get_traversal_range(max_traversal_type, max_traversal, mean=0, std=1):
    """
    Get the range of traversal for a latent variable.

    Parameters:
    - max_traversal_type (str): Type of traversal ('absolute' or 'probability').
    - max_traversal (float): Maximum traversal value.
    - mean (float): Mean of the latent variable.
    - std (float): Standard deviation of the latent variable.

    Returns:
    - range (tuple): Range of traversal.
    """
    if max_traversal_type == 'absolute':
        return (-max_traversal, max_traversal)
    elif max_traversal_type == 'probability':
        return stats.norm.ppf((1 - max_traversal) / 2, loc=mean, scale=std), stats.norm.ppf((1 + max_traversal) / 2, loc=mean, scale=std)
    else:
        raise ValueError("Invalid max_traversal_type. Choose 'absolute' or 'probability'.")

def traverse_single_latent(vae_model,
                           latent_idx,
                           num_samples=10,
                           max_traversal_type='probability',
                           max_traversal=0.95,
                           ref_img=None,
                           use_ref_img_lat_std=False  # Add parameter
                           ):
    """
    Traverse a single latent variable.

    Parameters:
    - vae_model: The VAE model.
    - latent_idx (int): Index of the latent variable to traverse.
    - num_samples (int): Number of samples to generate.
    - max_traversal_type (str): Type of traversal ('absolute' or 'probability').
    - max_traversal (float): Maximum traversal value.
    - ref_img: Reference image for latent statistics.
    - use_ref_img_lat_std (bool): Whether to use reference image latent standard deviation.

    Returns:
    - traversals (list): List of traversed latent variable samples.
    """
    latent_mean = 0
    latent_std = 1

    if ref_img is not None and use_ref_img_lat_std:
        ref_latent = vae_model.encode(ref_img)
        latent_mean = ref_latent.mean[latent_idx].item()
        latent_std = ref_latent.stddev[latent_idx].item()

    traversal_range = get_traversal_range(max_traversal_type, max_traversal, mean=latent_mean, std=latent_std)
    traversal_values = torch.linspace(traversal_range[0], traversal_range[1], num_samples)

    traversals = []
    for value in traversal_values:
        latent_sample = torch.zeros(vae_model.latent_dim)
        latent_sample[latent_idx] = value
        traversals.append(vae_model.decode(latent_sample.unsqueeze(0)))

    return traversals

def traverse_all_latents(vae_model,
                         num_samples=10,
                         max_traversal_type='probability', 
                         max_traversal=0.95, 
                         ref_img=None,
                         use_ref_img_lat_std=False  # Add parameter
                         ):
    """
    Traverse all latent variables.

    Parameters:
    - vae_model: The VAE model.
    - num_samples (int): Number of samples to generate.
    - max_traversal_type (str): Type of traversal ('absolute' or 'probability').
    - max_traversal (float): Maximum traversal value.
    - ref_img: Reference image for latent statistics.
    - use_ref_img_lat_std (bool): Whether to use reference image latent standard deviation.

    Returns:
    - all_traversals (list): List of traversals for all latent variables.
    """
    all_traversals = []
    for latent_idx in range(vae_model.latent_dim):
        traversals = traverse_single_latent(vae_model, latent_idx, num_samples, max_traversal_type, max_traversal, ref_img, use_ref_img_lat_std)
        all_traversals.append(traversals)

    return all_traversals