import os
import uuid
import logging
import csv
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra # type: ignore
from hydra.core.config_store import ConfigStore # type: ignore
from omegaconf import DictConfig, OmegaConf # type: ignore
import pandas as pd
import numpy as np

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

# Import core modules
from trainers.basetrainer import BaseTrainer
from utils.reproducibility import set_deterministic_run
from utils.helpers import get_model_device
from utils.io import find_optimal_num_workers
import datasets.utils as dataset_utils
import vae_models
import losses
from metrics.utils import MetricAggregator

# Setup console logging (file logging will be added per experiment)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manages experiment-based training with multiple seeds."""
    
    def __init__(self, experiment_config):
        self.experiment_config = experiment_config  # This is the full ExperimentConfig
        self.experiment_id = experiment_config.experiment_id or self._generate_experiment_id()
        
        # Ensure the base experiments directory exists
        base_experiments_dir = Path(experiment_config.results_dir)
        base_experiments_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Base experiments directory: {base_experiments_dir.absolute()}")
        
        # Create the specific experiment directory
        self.results_dir = base_experiments_dir / self.experiment_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Experiment directory created: {self.results_dir.absolute()}")
        
        # Create subdirectories
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # File paths
        self.results_csv_path = self.results_dir / "experiment_results.csv"
        self.summary_csv_path = self.results_dir / "experiment_summary.csv"
        self.config_path = self.results_dir / "experiment_config.yaml"
        self.completed_seeds_path = self.results_dir / "completed_seeds.json"
        self.log_file_path = self.results_dir / "experiment.log"
        
        # Setup experiment-specific logging
        self._setup_experiment_logging()
        
        # Save experiment configuration
        self._save_experiment_config()
        
        logger.info(f"Experiment {self.experiment_id} initialized")
    
    def _setup_experiment_logging(self):
        """Setup experiment-specific logging to both console and file."""
        # Get root logger
        root_logger = logging.getLogger()
        
        # Ensure console logging is properly configured
        if not any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) 
                  for handler in root_logger.handlers):
            # Add console handler if not present
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # Create file handler for experiment log
        file_handler = logging.FileHandler(self.log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter for log file (more detailed than console)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add file handler to root logger
        root_logger.addHandler(file_handler)
        
        # Store file handler reference for cleanup
        self.file_handler = file_handler
        
        # Set root logger level to ensure all messages are captured
        root_logger.setLevel(logging.INFO)
        
        # Log experiment initialization info (will go to both console and file)
        logger.info(f"Dual logging enabled - Console + File: {self.log_file_path}")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Base experiment directory: {self.results_dir.parent}")
        logger.info(f"Experiment directory: {self.results_dir}")
        logger.info(f"Number of seeds: {len(self.experiment_config.seeds)}")
        logger.info(f"Seeds: {self.experiment_config.seeds}")
        logger.info(f"Resume enabled: {self.experiment_config.resume}")
        logger.info("-" * 80)
    
    def cleanup_logging(self):
        """Clean up experiment-specific logging handlers."""
        if hasattr(self, 'file_handler'):
            # Log final experiment summary to both console and file
            logger.info("-" * 80)
            logger.info("EXPERIMENT COMPLETED")
            logger.info(f"Experiment ID: {self.experiment_id}")
            logger.info(f"Log file: {self.log_file_path}")
            logger.info(f"Results directory: {self.results_dir}")
            logger.info(f"Summary CSV: {self.summary_csv_path}")
            logger.info("-" * 80)
            
            # Remove file handler from root logger
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.file_handler)
            self.file_handler.close()
            
            # Log to console only that experiment logging is complete
            console_logger = logging.getLogger(__name__)
            console_logger.info(f"Dual logging completed. Console continues, file saved to: {self.log_file_path}")

    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}_{str(uuid.uuid4())[:8]}"
    
    def _save_experiment_config(self):
        """Save the experiment configuration."""
        with open(self.config_path, 'w') as f:
            OmegaConf.save(config=self.experiment_config, f=f)
    
    def get_completed_seeds(self) -> List[int]:
        """Get list of seeds that have already been completed."""
        if not self.completed_seeds_path.exists():
            return []
        
        with open(self.completed_seeds_path, 'r') as f:
            return json.load(f)
    
    def mark_seed_completed(self, seed: int):
        """Mark a seed as completed."""
        completed_seeds = self.get_completed_seeds()
        if seed not in completed_seeds:
            completed_seeds.append(seed)
            with open(self.completed_seeds_path, 'w') as f:
                json.dump(completed_seeds, f)
    
    def get_pending_seeds(self) -> List[int]:
        """Get list of seeds that still need to be run."""
        completed_seeds = self.get_completed_seeds()
        return [seed for seed in self.experiment_config.seeds if seed not in completed_seeds]
    
    def save_seed_results(self, seed: int, model: torch.nn.Module, dataset: torch.utils.data.Dataset, 
                         metric_config: Optional[MetricAggregatorConfig] = None, device: str = 'cpu'):
        """Save results for a specific seed by computing metrics using MetricAggregator."""
        
        # Initialize row data with basic info
        row_data = {
            'experiment_id': self.experiment_id,
            'seed': seed,
        }
        
        # Compute metrics if configuration is provided
        if metric_config is not None and metric_config.metrics:
            logger.info(f"Computing metrics for seed {seed}...")
            
            # Convert metric config to the format expected by MetricAggregator
            metrics_list = []
            for metric_cfg in metric_config.metrics:
                metric_dict = {
                    'name': metric_cfg.name,
                    'args': {k: v for k, v in metric_cfg.__dict__.items() if k != 'name'}
                }
                metrics_list.append(metric_dict)
            
            # Create MetricAggregator
            metric_aggregator = MetricAggregator(metrics_list)
            
            try:
                # Compute metrics using experiment seed for reproducible sampling
                computed_metrics = metric_aggregator.compute(
                    model=model, 
                    dataset=dataset, 
                    sample_num=metric_config.sample_num, 
                    seed=seed,  # Use the experiment seed for consistent sampling
                    device=device
                )
                
                # Add computed metrics to row data
                for metric_name, metric_value in computed_metrics.items():
                    if isinstance(metric_value, dict):
                        # If metric returns a dictionary, flatten it
                        for sub_key, sub_value in metric_value.items():
                            row_data[f'{metric_name}_{sub_key}'] = sub_value
                    else:
                        row_data[metric_name] = metric_value
                
                logger.info(f"Successfully computed {len(computed_metrics)} metrics for seed {seed}")
                
            except Exception as e:
                logger.error(f"Failed to compute metrics for seed {seed}: {str(e)}")
                # Add error information
                row_data['metric_computation_error'] = str(e)
        else:
            logger.warning(f"No metric configuration provided for seed {seed}, skipping metric computation")
        
        # Write to CSV
        file_exists = self.results_csv_path.exists()
        with open(self.results_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
            
        logger.info(f"Saved results for seed {seed} to {self.results_csv_path}")
        return row_data
    
    def generate_experiment_summary(self):
        """Generate final experiment summary with statistics across all seeds."""
        if not self.results_csv_path.exists():
            logger.warning("No results CSV file found for summary.")
            return

        # Load all results from the single CSV file
        df = pd.read_csv(self.results_csv_path)
        
        if df.empty:
            logger.warning("No completed seed results found for summary")
            return
        
        # Calculate statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        summary_data = {
            'experiment_id': self.experiment_id,
            'total_seeds': len(self.experiment_config.seeds),
            'completed_seeds': len(df),
            'summary_timestamp': datetime.now().isoformat()
        }
        
        # Add mean and std for each numeric metric
        for col in numeric_columns:
            if col not in ['seed', 'experiment_id', 'training_completed']:
                summary_data[f'{col}_mean'] = df[col].mean()
                summary_data[f'{col}_std'] = df[col].std()
                summary_data[f'{col}_min'] = df[col].min()
                summary_data[f'{col}_max'] = df[col].max()
        
        # Save summary
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_csv(self.summary_csv_path, index=False)
        
        logger.info(f"Generated experiment summary: {self.summary_csv_path}")
        logger.info(f"Completed {len(df)}/{len(self.experiment_config.seeds)} seeds")
        
        return summary_data
    
    def get_checkpoint_path(self, seed: int) -> str:
        """Get checkpoint path for a specific seed."""
        return str(self.checkpoints_dir / f"seed_{seed}")
    
    def get_log_file_path(self) -> str:
        """Get the path to the experiment log file."""
        return str(self.log_file_path)
    
    def verify_dual_logging(self) -> bool:
        """Verify that both console and file logging are active."""
        root_logger = logging.getLogger()
        
        # Check for console handler
        has_console = any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) 
                         for handler in root_logger.handlers)
        
        # Check for file handler
        has_file = hasattr(self, 'file_handler') and self.file_handler in root_logger.handlers
        
        logger.info(f"Logging verification - Console: {has_console}, File: {has_file}")
        return has_console and has_file


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Run experiment-based training with multiple seeds.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Dictionary containing experiment results
    """
    # Initialize experiment manager
    exp_manager = ExperimentManager(cfg)
    
    # Verify dual logging is working
    exp_manager.verify_dual_logging()
    
    # Get pending seeds (supports resume functionality)
    pending_seeds = exp_manager.get_pending_seeds()
    completed_seeds = exp_manager.get_completed_seeds()
    
    if completed_seeds:
        logger.info(f"Resuming experiment. Already completed seeds: {completed_seeds}")
    
    if not pending_seeds:
        logger.info("All seeds have been completed for this experiment!")
        summary = exp_manager.generate_experiment_summary()
        # Clean up experiment logging
        exp_manager.cleanup_logging()
        return {'experiment_summary': summary, 'experiment_id': exp_manager.experiment_id}
    
    logger.info(f"Running experiment {exp_manager.experiment_id}")
    logger.info(f"Experiment log file: {exp_manager.get_log_file_path()}")
    logger.info(f"Pending seeds: {pending_seeds}")
    logger.info("All training logs will be written to both console and experiment log file")
    
    experiment_results = {
        'experiment_id': exp_manager.experiment_id,
        'seed_results': {}
    }
    
    # Run training for each pending seed
    for seed in pending_seeds:
        logger.info("="*70)
        logger.info(f"Starting training for seed {seed}")
        train_id = f"{exp_manager.experiment_id}_seed_{seed}"
        logger.info(f"Seed {seed} - Training session: {train_id}")
        logger.info("="*70)
        
        try:
            # Create seed-specific configuration
            seed_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            
            # Set up seed-specific checkpoint directory
            seed_checkpoint_dir = exp_manager.get_checkpoint_path(seed)
            seed_cfg.trainer.checkpoint.enabled = True
            seed_cfg.trainer.checkpoint.save_master_dir = seed_checkpoint_dir
            
            # Run training for this seed
            training_output = run_training_session(seed_cfg.trainer, train_id, seed)
            
            # Extract components from training output
            model = training_output['model']
            dataset = training_output['dataset']
            device = training_output['device']
            
            # Get metric configuration if available
            metric_config = getattr(seed_cfg.trainer, 'metrics', None)
            
            # Save results with metric computation
            row_data = exp_manager.save_seed_results(
                seed=seed, 
                model=model, 
                dataset=dataset, 
                metric_config=metric_config,
                device=device
            )
            experiment_results['seed_results'][seed] = row_data
            
            # Mark seed as completed
            exp_manager.mark_seed_completed(seed)
            
            logger.info(f"Successfully completed training for seed {seed}")
            logger.info(f"Seed {seed} results saved to: {exp_manager.results_csv_path}")
            
        except Exception as e:
            logger.error(f"Training failed for seed {seed}: {str(e)}")
            logger.error("Experiment terminated due to training failure")
            # Clean up experiment logging before exiting
            exp_manager.cleanup_logging()
            raise
    
    # Generate final experiment summary
    logger.info("="*70)
    logger.info("Generating experiment summary...")
    logger.info("="*70)
    
    summary = exp_manager.generate_experiment_summary()
    experiment_results['experiment_summary'] = summary
    
    logger.info(f"Experiment {exp_manager.experiment_id} completed!")
    
    # Clean up experiment logging
    exp_manager.cleanup_logging()
    
    return experiment_results


def run_training_session(trainer_cfg: TrainerConfig, train_id: str, seed: int) -> Dict[str, Any]:
    """
    Run a single training session.
    
    Args:
        trainer_cfg: Trainer configuration object
        train_id: Unique identifier for the training run
        seed: Seed for reproducibility
        
    Returns:
        Dictionary containing training results
    """
    # Setup reproducibility first
    setup_reproducibility(trainer_cfg.determinism, seed)
    
    logger.info(f"Training ID: {train_id}")
    
    # Create dataset
    dataset = create_dataset(trainer_cfg.dataset)
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Get image size from dataset
    img_size = dataset_utils.get_img_size(trainer_cfg.dataset.name, getattr(trainer_cfg.dataset, 'img_size', None))
    logger.info(f"Image size: {img_size}")
    
    # Setup device
    device = setup_device(trainer_cfg.model)
    trainer_cfg.model.device = device  # Update config with actual device
    trainer_cfg.model.img_size = img_size  # Update config with actual image size
    
    # Create model
    model = create_model(trainer_cfg.model, img_size)
    model = model.to(device)
    logger.info(f"Model device: {get_model_device(model)}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss
    loss = create_loss(trainer_cfg.loss)
    
    # Create optimizer
    optimizer = create_optimizer(trainer_cfg.optimizer, model)
    
    # Create learning rate scheduler
    lr_scheduler = create_lr_scheduler(trainer_cfg.lr_scheduler, optimizer)
    
    # Create dataloader
    dataloader = create_dataloader(dataset, trainer_cfg.dataloader)
    logger.info(f"DataLoader: {len(dataloader)} batches, batch_size={trainer_cfg.dataloader.batch_size}")
    
    # Setup checkpointing
    checkpoint_kwargs = setup_checkpointing(trainer_cfg.checkpoint)
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = BaseTrainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_id=train_id,
        determinism_kwargs=OmegaConf.to_container(trainer_cfg.determinism, resolve=True),
        use_torch_compile=trainer_cfg.torch_compile.enabled,
        torch_compile_kwargs={
            'mode': trainer_cfg.torch_compile.mode,
            'backend': trainer_cfg.torch_compile.backend,
            'fullgraph': trainer_cfg.torch_compile.fullgraph,
            'dynamic': trainer_cfg.torch_compile.dynamic
        },
        dataloader=dataloader,
        # Progress bar
        is_progress_bar=trainer_cfg.progress_bar.enabled,
        progress_bar_log_iter_interval=trainer_cfg.progress_bar.log_iter_interval,
        # Logging
        use_train_logging=trainer_cfg.logging.enabled,
        return_logs=trainer_cfg.logging.return_logs,
        log_loss_interval_type=trainer_cfg.logging.loss_interval_type,
        log_loss_iter_interval=trainer_cfg.logging.loss_iter_interval,
        prev_train_losses_log=trainer_cfg.logging.prev_train_losses_log,
        log_metrics_interval_type=trainer_cfg.logging.metrics_interval_type,
        log_metrics_iter_interval=trainer_cfg.logging.metrics_iter_interval,
        prev_train_metrics_log=trainer_cfg.logging.prev_train_metrics_log,
        # Checkpointing
        **checkpoint_kwargs
    )
    
    # Start training
    logger.info(f"Starting training for {trainer_cfg.max_steps} {trainer_cfg.step_unit}(s)...")
    
    try:
        results = trainer.train(
            step_unit=trainer_cfg.step_unit,
            max_steps=trainer_cfg.max_steps,
            dataloader=dataloader
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Final results: {len(results['logs']['train_losses_log'])} loss entries, "
                   f"{len(results['logs']['train_metrics_log'])} metric entries")
        
        if trainer_cfg.checkpoint.return_chkpt:
            logger.info(f"Generated {len(results['chkpts'])} checkpoints")
        
        # Return results along with trained model and dataset for metric computation
        return {
            'training_results': results,
            'model': model,
            'dataset': dataset,
            'device': device
        }
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


def create_dataset(cfg: DatasetConfig) -> torch.utils.data.Dataset:
    """Create dataset from configuration."""
    logger.info(f"Creating {cfg.name} dataset from {cfg.root}")
    
    dataset_class = dataset_utils.get_dataset(cfg.name)
    
    # Convert config to dict
    dataset_kwargs = OmegaConf.to_container(cfg, resolve=True)
    dataset_kwargs.pop('name', None)  # Remove name as it's not needed for instantiation
    
    return dataset_class(**dataset_kwargs)


def create_model(cfg: ModelConfig, img_size: tuple) -> torch.nn.Module:
    """Create model from configuration."""
    logger.info(f"Creating {cfg.name} model")
    
    # Convert config to dict for model creation
    model_kwargs = OmegaConf.to_container(cfg, resolve=True)
    model_kwargs['img_size'] = img_size
    
    return vae_models.select(**model_kwargs)


def create_loss(cfg: LossConfig) -> torch.nn.Module:
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


def setup_reproducibility(determinism_cfg, seed: int) -> None:
    """Setup reproducibility based on configuration and experiment seed."""
    logger.info(f"Setting up reproducibility with seed: {seed}")
    set_deterministic_run(
        seed=seed,
        use_cuda_det=determinism_cfg.use_cuda_det,
        enforce_det=determinism_cfg.enforce_det
    )


def setup_device(cfg: ModelConfig) -> str:
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
def main(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Main entry point - handles experiment-based training.
    User is responsible for configuring experiments properly.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Dictionary containing training results or experiment summary
    """
    logger.info("="*50)
    if len(cfg.seeds) == 1:
        logger.info("Starting Single Training Session (Experiment Mode)")
        logger.info(f"Seed: {cfg.seeds[0]}")
    else:
        logger.info("Starting Multi-Seed Experiment")
        logger.info(f"Seeds: {cfg.seeds}")
    logger.info("="*50)
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Test dual logging
    logger.info("This message should appear in both console and experiment log file")
    
    # Run experiment
    return run_experiment(cfg)


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