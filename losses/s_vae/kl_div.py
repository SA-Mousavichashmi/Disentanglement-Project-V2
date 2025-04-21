import torch
import math
from power_spherical import PowerSpherical, HypersphericalUniform
import torch.distributions as D

def kl_power_spherical_uniform_loss(latent_factors_dist_param, return_components=False):
    """
    Calculates the KL divergence between Power Spherical posterior distributions q(z|x)
    and a uniform prior p(z) on the hypersphere S^(D-1) for multiple latent factors.

    The KL divergence for a single factor is computed using torch.distributions.kl.kl_divergence
    between the PowerSpherical distribution q(z|x) parameterized by mu and kappa, and a
    HypersphericalUniform distribution p(z). The total KL divergence is the sum of the
    KL divergences for each latent factor, averaged over the batch dimension.

    Parameters
    ----------
    latent_factors_dist_param : list[torch.Tensor]
        A list where each element is a tensor representing the parameters for the
        Power Spherical distribution of a single latent factor.
        Each tensor has shape (batch_size, D + 1), where the first D elements
        represent the location parameter `mu` (on the hypersphere S^(D-1)) and
        the last element represents the concentration parameter `kappa`.
        The length of the list corresponds to the number of latent factors.
    return_components : bool, optional
        If True, returns the average KL divergence for each latent factor separately.
        If False (default), returns the sum of the average KL divergences across all factors.

    Returns
    -------
    torch.Tensor
        - If return_components is False: A scalar tensor representing the total KL divergence,
          summed across all latent factors and averaged over the batch.
        - If return_components is True: A tensor of shape (latent_factor_num,) containing
          the KL divergence for each factor, averaged over the batch.
    """
    kl_components = []

    for params in latent_factors_dist_param:
        # Assuming the last element is kappa and the rest is mu
        mu = params[:, :-1]  # Shape (batch_size, D)
        kappa = params[:, -1] # Shape (batch_size,)
        
        # Ensure kappa is positive and non-zero for stability
        # kappa = torch.clamp(kappa, min=1e-6) 

        q_z_x = PowerSpherical(mu, kappa)
        
        # Determine dimension D for the hypersphere
        dim = mu.shape[-1]
        p_z = HypersphericalUniform(dim, device=mu.device)
        
        # KL divergence per batch element for this factor, then averaged
        kl_factor = D.kl.kl_divergence(q_z_x, p_z).mean()
        
        kl_components.append(kl_factor)

    kl_components_tensor = torch.stack(kl_components) # Shape (latent_factor_num,)

    if return_components:
        return kl_components_tensor
    else:
        return kl_components_tensor.sum() # Total KL divergence summed over factors
