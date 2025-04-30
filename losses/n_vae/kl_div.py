import torch
import torch.nn.functional as F


def kl_normal_loss(mean, logvar, return_components=False, raw=False):
    """
    Calculates the Kullback-Leibler (KL) divergence between a diagonal-covariance normal distribution
    and a unit normal distribution (0 mean, identity covariance). This is commonly used in VAEs
    to regularize latent space distributions toward isotropic unit Gaussians.

    The KL divergence is computed analytically using the formula:
    KL = 0.5 * ∑(1 + logσ² - μ² - σ²) where μ=mean, σ²=exp(logvar)

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim)
        latent_dim is the dimensionality of the latent space.

    logvar : torch.Tensor
        Log variance of the normal distribution. Shape (batch_size, latent_dim)
        Must be the natural logarithm of variance values.

    return_components : bool, default=False
        If True, returns KL values for each latent dimension separately.
        If False, returns the summed KL value across all dimensions.

    raw : bool, default=False
        If True, returns unaggregated KL values (same as return_components=True).
        This is a legacy parameter maintained for backward compatibility.

    Returns
    -------
    torch.Tensor
        If raw=True: tensor of shape (batch_size, latent_dim) containing per-dimension KL values for each batch item.
        If return_components=True (and raw=False): tensor of shape (latent_dim,) containing batch-averaged per-dimension KL values.
        Otherwise (raw=False and return_components=False): scalar tensor representing
        total KL divergence across all latent dimensions and averaged over the batch.

    Notes
    -----
    This function implements the standard KL divergence formula used in variational
    inference for diagonal Gaussian posteriors. The implementation uses numerically
    stable operations through PyTorch's tensor operations.
    """
    # KL divergence for each latent dimension and each batch item
    latent_kl_components = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp())

    if raw:
       # Return per-batch, per-component KL values
       return latent_kl_components
    else:
        # Batch mean of KL for each latent dimension
        latent_kl_mean_components = latent_kl_components.mean(dim=0)
        if return_components:
            # Return batch-averaged per-component KL values
            return latent_kl_mean_components
        # Return total KL divergence averaged over batch and summed over components
        return latent_kl_mean_components.sum()

def kl_divergence(mean_1, mean_2, logvar_1, logvar_2):
  var_1 = torch.exp(logvar_1)
  var_2 = torch.exp(logvar_2)
  return 1/2 * (var_1/var_2 + torch.square(mean_2-mean_1)/var_2 - 1 + logvar_2 - logvar_1)