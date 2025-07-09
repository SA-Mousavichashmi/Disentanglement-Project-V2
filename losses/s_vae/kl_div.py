import torch
import math
from power_spherical import PowerSpherical, HypersphericalUniform # type: ignore
import torch.distributions as D

def kl_power_spherical_uniform_factor_wise(params, reduction='mean'):
    """
    Calculates KL divergence for a single latent factor between Power Spherical posterior
    and uniform prior on the hypersphere.

    Parameters
    ----------
    params : torch.Tensor
        Tensor of shape (batch_size, D + 1), where:
            - params[:, :-1] (shape: (batch_size, D)) are the location parameters `mu`
            - params[:, -1] (shape: (batch_size,)) is the concentration parameter `kappa`
        `D` is the dimension of the hypersphere (i.e., latent space dimension per factor).
    reduction : str, optional
        Specifies the reduction to apply to the output:
        - 'mean': returns the mean of KL divergences over the batch
        - 'sum': returns the sum of KL divergences over the batch  
        - 'none': returns the raw KL divergences for each sample in the batch

    Returns
    -------
    torch.Tensor
        KL divergence tensor:
        - If reduction='mean': scalar tensor (average over batch)
        - If reduction='sum': scalar tensor (sum over batch)
        - If reduction='none': tensor of shape (batch_size,) with per-sample KL divergences
    """
    mu = params[:, :-1]
    kappa = params[:, -1]
    q_z_x = PowerSpherical(mu, kappa)
    dim = mu.shape[-1]
    p_z = HypersphericalUniform(dim, device=mu.device)
    kl_divs = D.kl.kl_divergence(q_z_x, p_z)
    
    if reduction == 'mean':
        return kl_divs.mean()
    elif reduction == 'sum':
        return kl_divs.sum()
    elif reduction == 'none':
        return kl_divs
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Must be 'mean', 'sum', or 'none'.")

def kl_power_spherical_uniform_loss(latent_factors_dist_param, return_components=False, reduction='mean'):
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
        If True, returns the KL divergence for each latent factor separately.
        If False (default), returns the sum of the KL divergences across all factors.
    reduction : str, optional
        Specifies the reduction to apply to the output:
        - 'mean': returns the mean of KL divergences over the batch
        - 'sum': returns the sum of KL divergences over the batch  
        - 'none': returns the raw KL divergences for each sample in the batch

    Returns
    -------
    torch.Tensor
        - If return_components is False: KL divergence summed across all latent factors
        - If return_components is True: A tensor containing the KL divergence for each factor
        The shape depends on the reduction parameter:
        - reduction='mean' or 'sum': scalar or (latent_factor_num,) respectively
        - reduction='none': (batch_size,) or (batch_size, latent_factor_num) respectively
    """
    kl_components = []

    for params in latent_factors_dist_param:
        kl_factor = kl_power_spherical_uniform_factor_wise(params, reduction=reduction)
        kl_components.append(kl_factor)

    if reduction == 'none':
        # Stack along factor dimension: (batch_size, latent_factor_num)
        kl_components_tensor = torch.stack(kl_components, dim=1)
    else:
        # Stack into (latent_factor_num,) tensor
        kl_components_tensor = torch.stack(kl_components)

    if return_components:
        return kl_components_tensor
    else:
        if reduction == 'none':
            return kl_components_tensor.sum(dim=1)  # Sum over factors, keep batch dimension
        else:
            return kl_components_tensor.sum()  # Total KL divergence summed over factors
