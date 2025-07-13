import torch
from .basemetric import BaseMetric
from losses.n_vae.kl_div import kl_normal_loss


class KLD(BaseMetric):
    def __init__(self, reduction="sum", **kwargs):
        """
        Initialize KLD metric for computing KL divergence to unit normal.
        
        Parameters
        ----------
        reduction : str, optional
            Method for aggregating KL divergences across the batch dimension.
            Options: "sum" or "mean". Defaults to "sum".
            - "sum": Each KL_i is the sum of KL values across the batch for dimension i
            - "mean": Each KL_i is the mean of KL values across the batch for dimension i
        """
        super().__init__(**kwargs)
        if reduction not in ["sum", "mean"]:
            raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction}")
        self.reduction = reduction

    @property
    def _requires(self):
        return ['stats_qzx']

    @property
    def _mode(self):
        return 'batch'

    def __call__(self, stats_qzx, **kwargs):
        """
        Compute KL-Divergence to unit normal for n_vae models.
        
        Parameters
        ----------
        stats_qzx : torch.Tensor
            Encoder statistics tensor of shape (batch_size, latent_dim, 2) 
            containing mean and logvar.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'KL_i': KL divergence for dimension i (aggregated across batch based on reduction method)
            - 'KL': Total KL divergence (sum of all dimension-wise KLs)
        """
        # Validate input format
        if not isinstance(stats_qzx, torch.Tensor):
            raise ValueError(f"stats_qzx must be a torch.Tensor, got {type(stats_qzx)}")
        
        if stats_qzx.dim() != 3 or stats_qzx.size(-1) != 2:
            raise ValueError(f"Expected stats_qzx tensor to have shape (batch_size, latent_dim, 2), got {stats_qzx.shape}")
        
        # Extract mean and logvar from tensor
        mean, logvar = stats_qzx.unbind(-1)
        
        # Compute raw KL divergences: (batch_size, latent_dim)
        kl_raw = kl_normal_loss(mean, logvar, raw=True)
        
        # Apply reduction across batch dimension
        if self.reduction == "sum":
            kl_components = kl_raw.sum(dim=0)  # Sum across batch
        elif self.reduction == "mean":
            kl_components = kl_raw.mean(dim=0)  # Mean across batch
        
        # Create output dictionary
        result = {}
        
        # Add individual dimension KLs
        for i, kl_val in enumerate(kl_components):
            result[f'KL_{i}'] = kl_val.item()
        
        # Add total KL (always sum of dimension-wise KLs)
        result['KL'] = kl_components.sum().item()
        
        return result