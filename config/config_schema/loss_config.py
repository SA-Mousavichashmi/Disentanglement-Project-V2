"""
Configuration schema for all loss functions used in the codebase.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from omegaconf import MISSING # type: ignore


@dataclass
class LossConfig:
    """Base class for loss configuration."""
    name: str = MISSING
    log_kl_components: bool = False
    rec_dist: str = "bernoulli"  # Default reconstruction distribution
    schedulers_kwargs: Optional[List[Dict[str, Any]]] = None

########## R1 * .... * R1 latent topology ##########

@dataclass
class BetaVAEConfig(LossConfig):
    """Configuration for Beta Variational Autoencoder loss."""
    name: str = "betavae"
    beta: float = 4.0

@dataclass
class AnnealedVAEConfig(LossConfig):
    """Configuration for Annealed Variational Autoencoder loss."""
    name: str = "annealedvae"
    gamma: float = 100.0
    C_init: float = 0.0
    C_fin: float = 5.0
    anneal_steps: int = 100000

@dataclass
class FactorVAEConfig(LossConfig):
    """Configuration for Factorized Variational Autoencoder loss."""
    name: str = "factorvae"
    gamma: float = 10.0
    discr_lr: float = 1e-4
    discr_betas: tuple = (0.5, 0.9)
    external_optimization: bool = False

@dataclass
class BetaTCVAEConfig(LossConfig):
    """Configuration for Beta-TCVAE loss."""
    name: str = "betatcvae"
    alpha: float = 1.0
    beta: float = 6.0
    gamma: float = 1.0
    n_data: int = MISSING
    is_mss: bool = True


# DIP-VAE-I config
@dataclass
class DipVAEIConfig(LossConfig):
    """Configuration for DIP-VAE-I loss."""
    name: str = "dipvae-i"
    lambda_od: float = 10.0  # Weight for off-diagonal covariance penalty
    lambda_d: float = 100.0  # Weight for diagonal covariance penalty
    beta: float = 1.0  # Weight of the KL term

# DIP-VAE-II config
@dataclass
class DipVAEIIConfig(LossConfig):
    """Configuration for DIP-VAE-II loss."""
    name: str = "dipvae-ii"
    lambda_od: float = 10.0  # Weight for off-diagonal covariance penalty
    lambda_d: float = 100.0  # Weight for diagonal covariance penalty
    beta: float = 1.0  # Weight of the KL term

########## S1 * .... * S1 latent topology ##########

@dataclass
class BetaToroidalVAEConfig(LossConfig):
    """Configuration for Beta Toroidal Variational Autoencoder loss."""
    name: str = "beta_toroidal_vae"
    beta: float = 1.0

######## mixture of R1 and S1 latent topologies ##########

@dataclass
class BetaSNVAEConfig(LossConfig):
    """Configuration for Beta-SN Variational Autoencoder loss."""
    name: str = "beta_s_n_vae"
    beta: float = 1.0
    latent_factor_topologies: Optional[List[str]] = None

@dataclass
class AnnealSNVAEConfig(LossConfig):
    """Configuration for Annealed-SN Variational Autoencoder loss."""
    name: str = "anneal_s_n_vae"
    gamma: float = 100.0
    C_init: float = 0.0
    C_fin: float = 5.0
    anneal_steps: int = 100000
    latent_factor_topologies: Optional[List[str]] = None

########### Group Theory based loss ##########

@dataclass
class GANConfig:
    """Configuration for GAN components used in group theory loss (g-meaningful loss).

    This configuration class encapsulates all GAN-related parameters including:
    - Loss function type and parameters
    - Discriminator architecture and parameters  
    - Optimizer type and parameters
    """
    loss_type: str = "wgan_gp"  # Options: "wgan_gp", "hinge", "bce", "lsgan", "wgan"
    loss_kwargs: Optional[Dict[str, Any]] = None
    d_arch: str = "locatello"  # Options: "locatello", "spectral_norm"
    d_kwargs: Optional[Dict[str, Any]] = None
    d_optimizer_type: str = "adam"  # Options: "adam", "rmsprop", "sgd"
    d_optimizer_kwargs: Optional[Dict[str, Any]] = None

@dataclass
class GroupTheoryConfig(LossConfig):
    """Configuration for Group Theory based loss."""
    name: str = "group_theory"
    base_loss: LossConfig = field(default_factory=BetaVAEConfig)
    rec_dist: str = "bernoulli"
    device: str = "cpu"
    ## g-commutative loss parameters
    commutative_weight: float = 1.0
    commutative_component_order: int = 2
    commutative_comparison_dist: str = "gaussian"
    meaningful_weight: float = 1.0
    meaningful_component_order: int = 1
    ## g-meaningful loss parameters
    meaningful_transformation_order: int = 1
    meaningful_n_critic: int = 1
    meaningful_gan_config: GANConfig = field(default_factory=GANConfig)
    # general group theory parameters
    deterministic_rep: bool = True
    group_action_latent_range: float = 2.0
    group_action_latent_distribution: str = "uniform"
    comp_latent_select_threshold: float = 0.0
    base_loss_state_dict: Optional[Dict[str, Any]] = None
    warm_up_steps: int = 0
