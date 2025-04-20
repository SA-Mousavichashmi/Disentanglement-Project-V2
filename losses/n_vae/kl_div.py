import torch
import torch.nn.functional as F


def kl_normal_loss(mean, logvar, return_components=False):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)

    return_components: boolean
        Return loss for each latent dim.
    """
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    if return_components:
        return latent_kl
    return latent_kl.sum()

def kl_divergence(mean_1, mean_2, logvar_1, logvar_2):
  var_1 = torch.exp(logvar_1)
  var_2 = torch.exp(logvar_2)
  return 1/2 * (var_1/var_2 + torch.square(mean_2-mean_1)/var_2 - 1 + logvar_2 - logvar_1)