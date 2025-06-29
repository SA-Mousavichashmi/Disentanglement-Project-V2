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

@dataclass
class BetaTCVAEConfig(LossConfig):
    """Configuration for Beta-TCVAE loss."""
    name: str = "betatcvae"
    alpha: float = 1.0
    beta: float = 6.0
    gamma: float = 1.0
    n_data: Optional[int] = None
    is_mss: bool = True

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
class GroupTheoryConfig(LossConfig):
    """Configuration for Group Theory based loss."""
    name: str = "group_theory"
    base_loss_name: str = "betavae"
    base_loss_kwargs: Dict[str, Any] = field(default_factory=lambda: BetaVAEConfig().__dict__)
    rec_dist: str = "bernoulli"
    device: str = "cpu"
    commutative_weight: float = 1.0
    commutative_component_order: int = 2
    commutative_comparison_dist: str = "gaussian"
    meaningful_weight: float = 1.0
    meaningful_component_order: int = 1
    meaningful_transformation_order: int = 1
    meaningful_critic_gradient_penalty_weight: float = 10.0
    meaningful_critic_lr: float = 1e-4
    meaningful_n_critic: int = 5
    deterministic_rep: bool = True
    group_action_latent_range: float = 1.0
    group_action_latent_distribution: str = "normal"
    comp_latent_select_threshold: float = 0.0
    base_loss_state_dict: Optional[Dict[str, Any]] = None
    warm_up_steps: int = 0
