from typing import Optional, Dict, Any, List
import hydra # type: ignore
from hydra.core.config_store import ConfigStore # type: ignore
from omegaconf import DictConfig, OmegaConf # type: ignore

# Import configuration schemas
from config.config_schema.trainer_config import ExperimentConfig, TrainerConfig
from config.config_schema.dataset_config import (
    DatasetConfig, Cars3DConfig, DSpritesConfig, Shapes3DConfig
)
from config.config_schema.model_config import (
    ModelConfig, VAEConfig, VAEBurgessConfig, VAEChenMLPConfig, VAELocatelloConfig,
    VAELocatelloSBDConfig, VAEMonteroSmallConfig, VAEMonteroLargeConfig,
    ToroidalVAEConfig, ToroidalVAEBurgessConfig, ToroidalVAELocatelloConfig,
    SNVAEBurgessConfig, SNVAELocatelloConfig
)
from config.config_schema.loss_config import (
    LossConfig, BetaVAEConfig, AnnealedVAEConfig, FactorVAEConfig, BetaTCVAEConfig,
    BetaToroidalVAEConfig, BetaSNVAEConfig, AnnealSNVAEConfig, GroupTheoryConfig
)
from config.config_schema.metric_config import MetricAggregatorConfig


@hydra.main(version_base=None, config_path="config/configs", config_name="baseconfig")
def main(cfg: DictConfig) -> Dict[str, Any]:
    """
    Main entry point - handles experiment-based training.
    User is responsible for configuring experiments properly.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Dictionary containing training results or experiment summary
    """
    # Handle nested structure from config paths (e.g., experiments/config_name)
    if 'experiments' in cfg:
        # Get the nested config (e.g., cfg.experiments.anneal_vae_shapes3d)
        actual_cfg = cfg.experiments
        print(OmegaConf.to_yaml(actual_cfg))
        if hasattr(actual_cfg, 'seeds'):
            print(actual_cfg.seeds)
        return {"status": "success", "config": actual_cfg}
    else:
        # Standard configuration
        print(OmegaConf.to_yaml(cfg))
        if hasattr(cfg, 'seeds'):
            print(cfg.seeds)
        return {"status": "success", "config": cfg}

# Register configuration schemas with Hydra
def register_configs():
    """Register all configuration schemas with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    
    # Register main config
    cs.store(name="base_config", node=ExperimentConfig)
    
    # Register model configs
    cs.store(group="model", name="vae", node=VAEConfig)
    cs.store(group="model", name="vae_burgess", node=VAEBurgessConfig)
    cs.store(group="model", name="vae_chen_mlp", node=VAEChenMLPConfig)
    cs.store(group="model", name="vae_locatello", node=VAELocatelloConfig)
    cs.store(group="model", name="vae_locatello_sbd", node=VAELocatelloSBDConfig)
    cs.store(group="model", name="vae_montero_small", node=VAEMonteroSmallConfig)
    cs.store(group="model", name="vae_montero_large", node=VAEMonteroLargeConfig)
    cs.store(group="model", name="toroidal_vae", node=ToroidalVAEConfig)
    cs.store(group="model", name="toroidal_vae_burgess", node=ToroidalVAEBurgessConfig)
    cs.store(group="model", name="toroidal_vae_locatello", node=ToroidalVAELocatelloConfig)
    cs.store(group="model", name="s_n_vae_burgess", node=SNVAEBurgessConfig)
    cs.store(group="model", name="s_n_vae_locatello", node=SNVAELocatelloConfig)

    # Register loss configs
    cs.store(group="loss", name="betavae", node=BetaVAEConfig)
    cs.store(group="loss", name="annealedvae", node=AnnealedVAEConfig)
    cs.store(group="loss", name="factorvae", node=FactorVAEConfig)
    cs.store(group="loss", name="betatcvae", node=BetaTCVAEConfig)
    cs.store(group="loss", name="beta_toroidal_vae", node=BetaToroidalVAEConfig)
    cs.store(group="loss", name="beta_s_n_vae", node=BetaSNVAEConfig)
    cs.store(group="loss", name="anneal_s_n_vae", node=AnnealSNVAEConfig)
    cs.store(group="loss", name="group_theory", node=GroupTheoryConfig)

    # Register dataset configs
    cs.store(group="dataset", name="cars3d", node=Cars3DConfig)
    cs.store(group="dataset", name="dsprites", node=DSpritesConfig)
    cs.store(group="dataset", name="shapes3d", node=Shapes3DConfig)


if __name__ == "__main__":
    # Register configs before running
    register_configs()
    
    # Run main function
    main()
