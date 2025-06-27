"""
Configuration schema for all model architectures used in the codebase.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Union
from omegaconf import MISSING # type: ignore

@dataclass
class ModelConfig:
    """Base class for model configuration."""
    name: str = MISSING
    img_size: Tuple[int, int, int] = MISSING
    device: str = MISSING

@dataclass
class VAEBaseConfig(ModelConfig):
    encoder_name: str = MISSING
    decoder_name: str = MISSING
    decoder_output_dist: str = "bernoulli"
    use_batchnorm: bool = False

# ------------------- N-VAE Models -------------------

@dataclass
class NVAEConfig(VAEBaseConfig):
    latent_dim: int = 10

@dataclass
class VAEConfig(NVAEConfig):
    name: str = "vae"
    encoder_name: str = "locatello"
    decoder_name: str = "locatello"

@dataclass
class VAEBurgessConfig(NVAEConfig):
    name: str = "vae_burgess"
    encoder_name: str = "burgess"
    decoder_name: str = "burgess"

@dataclass
class VAEChenMLPConfig(NVAEConfig):
    name: str = "vae_chen_mlp"
    encoder_name: str = "chen_mlp"
    decoder_name: str = "chen_mlp"

@dataclass
class VAELocatelloConfig(NVAEConfig):
    name: str = "vae_locatello"
    encoder_name: str = "locatello"
    decoder_name: str = "locatello"
    encoder_decay: float = 0.0
    decoder_decay: float = 0.0

@dataclass
class VAELocatelloSBDConfig(NVAEConfig):
    name: str = "vae_locatello_sbd"
    encoder_name: str = "locatello_sbd"
    decoder_name: str = "locatello_sbd"

@dataclass
class VAEMonteroSmallConfig(NVAEConfig):
    name: str = "vae_montero_small"
    encoder_name: str = "montero_small"
    decoder_name: str = "montero_small"

@dataclass
class VAEMonteroLargeConfig(NVAEConfig):
    name: str = "vae_montero_large"
    encoder_name: str = "montero_large"
    decoder_name: str = "montero_large"

# ------------------- S-VAE Models -------------------

@dataclass
class SVAEConfig(VAEBaseConfig):
    latent_factor_num: int = 10

@dataclass
class ToroidalVAEConfig(SVAEConfig):
    name: str = "toroidal_vae"
    encoder_name: str = "chen_mlp"
    decoder_name: str = "chen_mlp"

@dataclass
class ToroidalVAEBurgessConfig(SVAEConfig):
    name: str = "toroidal_vae_burgess"
    encoder_name: str = "burgess"
    decoder_name: str = "burgess"

@dataclass
class ToroidalVAELocatelloConfig(SVAEConfig):
    name: str = "toroidal_vae_locatello"
    encoder_name: str = "locatello"
    decoder_name: str = "locatello"
    encoder_decay: float = 0.0
    decoder_decay: float = 0.0

# ------------------- S-N-VAE Models -------------------

@dataclass
class SNVAEConfig(VAEBaseConfig):
    latent_factor_topologies: List[str] = field(default_factory=lambda: ["R1", "S1", "R1"])

@dataclass
class SNVAEBurgessConfig(SNVAEConfig):
    name: str = "s_n_vae_burgess"
    encoder_name: str = "burgess"
    decoder_name: str = "burgess"
    
@dataclass
class SNVAELocatelloConfig(SNVAEConfig):
    name: str = "s_n_vae_locatello"
    encoder_name: str = "locatello"
    decoder_name: str = "locatello"
    encoder_decay: float = 0.0
    decoder_decay: float = 0.0

ModelConfigUnion = Union[
    VAEConfig,
    VAEBurgessConfig,
    VAEChenMLPConfig,
    VAELocatelloConfig,
    VAELocatelloSBDConfig,
    VAEMonteroSmallConfig,
    VAEMonteroLargeConfig,
    ToroidalVAEConfig,
    ToroidalVAEBurgessConfig,
    ToroidalVAELocatelloConfig,
    SNVAEBurgessConfig,
    SNVAELocatelloConfig
]
