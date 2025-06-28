import os
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra # type: ignore
from hydra.core.config_store import ConfigStore # type: ignore
from omegaconf import DictConfig, OmegaConf # type: ignore

# Import configuration schemas
from config.config_schema.trainer_config import BaseTrainerConfig
from config.config_schema.dataset_config import DatasetConfigUnion
from config.config_schema.model_config import ModelConfigUnion  
from config.config_schema.loss_config import LossConfigUnion
from config.config_schema.metric_config import MetricAggregatorConfig

# Import core modules
from trainers.basetrainer import BaseTrainer
from utils.reproducibility import set_deterministic_run
from utils.helpers import get_model_device
from utils.io import find_optimal_num_workers
import datasets.utils as dataset_utils
import vae_models
import losses

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dataset(cfg: DatasetConfigUnion) -> torch.utils.data.Dataset:
    """Create dataset from configuration."""
    logger.info(f"Creating {cfg.name} dataset from {cfg.root}")
    
    dataset_class = dataset_utils.get_dataset(cfg.name)
    
    # Convert config to dict
    dataset_kwargs = OmegaConf.to_container(cfg, resolve=True)
    
    return dataset_class(**dataset_kwargs)


def create_model(cfg: ModelConfigUnion, img_size: tuple) -> torch.nn.Module:
    """Create model from configuration."""
    logger.info(f"Creating {cfg.name} model")
    
    # Convert config to dict for model creation
    model_kwargs = OmegaConf.to_container(cfg, resolve=True)
    model_kwargs['img_size'] = img_size
    
    return vae_models.select(**model_kwargs)


def create_loss(cfg: LossConfigUnion) -> torch.nn.Module:
    """Create loss function from configuration."""
    logger.info(f"Creating {cfg.name} loss")
    
    # Convert config to dict for loss creation
    loss_kwargs = OmegaConf.to_container(cfg, resolve=True)
    
    return losses.select(**loss_kwargs)


def create_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create optimizer from configuration."""
    logger.info(f"Creating {cfg.name} optimizer with lr={cfg.lr}")
    
    if cfg.name.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps
        )
    elif cfg.name.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps
        )
    elif cfg.name.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=getattr(cfg, 'momentum', 0.9)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.name}")


def create_lr_scheduler(cfg, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from configuration."""
    if cfg is None or cfg.name is None:
        return None
        
    logger.info(f"Creating {cfg.name} learning rate scheduler")
    
    if cfg.name == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma,
            verbose=cfg.verbose
        )
    elif cfg.name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=cfg.patience,
            factor=cfg.factor,
            verbose=cfg.verbose
        )
    elif cfg.name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.eta_min,
            verbose=cfg.verbose
        )
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.name}")


def create_dataloader(dataset: torch.utils.data.Dataset, cfg) -> DataLoader:
    """Create DataLoader from dataset and configuration."""
    num_workers = cfg.num_workers
    if num_workers == -1:
        num_workers = find_optimal_num_workers(dataset=dataset, batch_size=cfg.batch_size, pin_memory=cfg.pin_memory)
        logger.info(f"Auto-detected optimal num_workers: {num_workers}")
    
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and num_workers > 0,
        drop_last=cfg.drop_last
    )


def setup_reproducibility(cfg) -> None:
    """Setup reproducibility based on configuration."""
    if cfg.seed is not None:
        logger.info(f"Setting up reproducibility with seed: {cfg.seed}")
        set_deterministic_run(
            seed=cfg.seed,
            use_cuda_det=cfg.use_cuda_det,
            enforce_det=cfg.enforce_det
        )
    else:
        logger.warning("No seed specified - training will not be reproducible")


def setup_device(cfg: ModelConfigUnion) -> str:
    """Setup device for training."""
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto-detected device: {device}")
    else:
        device = cfg.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
    
    logger.info(f"Using device: {device}")
    return device


def setup_checkpointing(cfg) -> Dict[str, Any]:
    """Setup checkpointing directories and paths."""
    checkpoint_kwargs = {}
    
    if cfg.enabled:
        if cfg.save_master_dir is not None:
            # Create organized checkpoint structure
            master_dir = Path(cfg.save_master_dir)
            master_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_kwargs['chkpt_save_master_dir'] = str(master_dir)
            
        elif cfg.save_dir is not None:
            # Create checkpoint directory
            save_dir = Path(cfg.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_kwargs['chkpt_save_dir'] = str(save_dir)
            
        elif cfg.save_path is not None:
            # Ensure parent directory exists
            save_path = Path(cfg.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_kwargs['chkpt_save_path'] = str(save_path)
        
        checkpoint_kwargs.update({
            'return_chkpt': cfg.return_chkpt,
            'chkpt_every_n_steps': cfg.every_n_steps,
            'chkpt_step_type': cfg.step_type,
            'chkpt_viz': cfg.save_viz
        })
    
    return checkpoint_kwargs


@hydra.main(version_base=None, config_path="config/examples", config_name="config")
def main(cfg: BaseTrainerConfig) -> Dict[str, Any]:
    """
    Main training function.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Dictionary containing training logs and optional checkpoints
    """
    logger.info("="*50)
    logger.info("Starting Disentanglement Learning Training")
    logger.info("="*50)
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Setup reproducibility first
    setup_reproducibility(cfg.determinism)
    
    # Generate training ID if not provided
    train_id = cfg.train_id or str(uuid.uuid4())
    logger.info(f"Training ID: {train_id}")
    
    # Create dataset
    dataset = create_dataset(cfg.dataset)
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Get image size from dataset
    img_size = dataset_utils.get_img_size(cfg.dataset.name, getattr(cfg.dataset, 'img_size', None))
    logger.info(f"Image size: {img_size}")
    
    # Setup device
    device = setup_device(cfg.model)
    cfg.model.device = device  # Update config with actual device
    cfg.model.img_size = img_size  # Update config with actual image size
    
    # Create model
    model = create_model(cfg.model, img_size)
    model = model.to(device)
    logger.info(f"Model device: {get_model_device(model)}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss
    loss = create_loss(cfg.loss)
    
    # Create optimizer  
    optimizer = create_optimizer(cfg.optimizer, model)
    
    # Create learning rate scheduler
    lr_scheduler = create_lr_scheduler(cfg.lr_scheduler, optimizer)
    
    # Create dataloader
    dataloader = create_dataloader(dataset, cfg.dataloader)
    logger.info(f"DataLoader: {len(dataloader)} batches, batch_size={cfg.dataloader.batch_size}")
    
    # Setup checkpointing
    checkpoint_kwargs = setup_checkpointing(cfg.checkpoint)
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = BaseTrainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_id=train_id,
        determinism_kwargs=OmegaConf.to_container(cfg.determinism, resolve=True),
        use_torch_compile=cfg.torch_compile.enabled,
        torch_compile_kwargs={
            'mode': cfg.torch_compile.mode,
            'backend': cfg.torch_compile.backend,
            'fullgraph': cfg.torch_compile.fullgraph,
            'dynamic': cfg.torch_compile.dynamic
        },
        prev_train_iter=cfg.prev_train_iter,
        dataloader=dataloader,
        # Progress bar
        is_progress_bar=cfg.progress_bar.enabled,
        progress_bar_log_iter_interval=cfg.progress_bar.log_iter_interval,
        # Logging
        use_train_logging=cfg.logging.enabled,
        return_logs=cfg.logging.return_logs,
        log_loss_interval_type=cfg.logging.loss_interval_type,
        log_loss_iter_interval=cfg.logging.loss_iter_interval,
        prev_train_losses_log=cfg.logging.prev_train_losses_log,
        log_metrics_interval_type=cfg.logging.metrics_interval_type,
        log_metrics_iter_interval=cfg.logging.metrics_iter_interval,
        prev_train_metrics_log=cfg.logging.prev_train_metrics_log,
        # Checkpointing
        **checkpoint_kwargs
    )
    
    # Start training
    logger.info(f"Starting training for {cfg.max_steps} {cfg.step_unit}(s)...")
    
    try:
        results = trainer.train(
            step_unit=cfg.step_unit,
            max_steps=cfg.max_steps,
            dataloader=dataloader
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Final results: {len(results['logs']['train_losses_log'])} loss entries, "
                   f"{len(results['logs']['train_metrics_log'])} metric entries")
        
        if cfg.checkpoint.return_chkpt:
            logger.info(f"Generated {len(results['chkpts'])} checkpoints")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


# Register configuration schemas with Hydra
def register_configs():
    """Register all configuration schemas with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    
    # Register main config
    cs.store(name="base_config", node=BaseTrainerConfig)
    
    # We could register specific configs here if needed
    # cs.store(group="dataset", name="cars3d", node=Cars3DConfig)
    # etc.


if __name__ == "__main__":
    # Register configs before running
    register_configs()
    
    # Run main function
    main()