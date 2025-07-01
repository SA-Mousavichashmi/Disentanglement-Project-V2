"""
Configuration schema for training models in the disentanglement project.
This serves as the master configuration that orchestrates all other configs.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
from omegaconf import MISSING # type: ignore
from .metric_config import MetricAggregatorConfig

@dataclass
class OptimizerConfig:
    """Configuration for optimizer settings."""
    name: str = "Adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class LRSchedulerConfig:
    """Configuration for learning rate scheduler."""
    name: Optional[str] = None  # None, "StepLR", "ReduceLROnPlateau", "CosineAnnealingLR", etc.
    step_size: int = 30  # For StepLR
    gamma: float = 0.1  # For StepLR and ReduceLROnPlateau
    patience: int = 10  # For ReduceLROnPlateau
    factor: float = 0.1  # For ReduceLROnPlateau
    T_max: int = 100  # For CosineAnnealingLR
    eta_min: float = 0  # For CosineAnnealingLR
    verbose: bool = False


@dataclass
class DeterminismConfig:
    """Configuration for reproducibility and determinism."""
    use_cuda_det: bool = True
    enforce_det: bool = False


@dataclass
class TorchCompileConfig:
    """Configuration for torch.compile optimization."""
    enabled: bool = False
    mode: str = "max-autotune"  # "default", "reduce-overhead", "max-autotune"
    backend: str = "inductor"  # "inductor", "aot_eager", "cudagraphs"
    fullgraph: bool = False
    dynamic: bool = False


@dataclass
class DataLoaderConfig:
    """Configuration for data loading."""
    batch_size: int = 64
    num_workers: int = 4 # if set -1 it will determine automatically based on available CPUs
    pin_memory: bool = True
    persistent_workers: bool = True
    shuffle: bool = True
    drop_last: bool = False


@dataclass
class ProgressBarConfig:
    """Configuration for progress bar display."""
    enabled: bool = True
    log_iter_interval: int = 50


@dataclass
class LoggingConfig:
    """Configuration for training logging."""
    enabled: bool = True
    return_logs: bool = True
    
    # Loss logging
    loss_interval_type: str = "iter"  # "iter" or "epoch"
    loss_iter_interval: int = 200
    prev_train_losses_log: Optional[List[Dict[str, Any]]] = None
    
    # Metrics logging
    metrics_interval_type: str = "iter"  # "iter" or "epoch"
    metrics_iter_interval: int = 200
    prev_train_metrics_log: Optional[List[Dict[str, Any]]] = None


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    enabled: bool = False
    return_chkpt: bool = False
    
    # Checkpoint frequency
    every_n_steps: Optional[int] = None  # None means only final checkpoint
    step_type: str = "iter"  # "iter" or "epoch"
    
    # Save locations (mutually exclusive)
    save_path: Optional[str] = None  # Single file path
    save_dir: Optional[str] = None  # Directory for multiple checkpoints
    save_master_dir: Optional[str] = None  # Master directory for organized checkpoints
    
    # Visualization
    save_viz: bool = False  # Save visualizations with checkpoints


@dataclass
class TrainerConfig:
    """
    Configuration for model training that orchestrates all core training components.
    This directly maps to the BaseTrainer constructor parameters.
    """        
    # ================ Training Progression ================
    step_unit: str = "epoch"  # "epoch" or "iter"
    max_steps: int = 100  # Number of epochs or iterations
    device: str = "auto" # "auto", "cpu", "cuda"
    
    # ================ Core Required Components ================
    model: Any = MISSING
    loss: Any = MISSING
    dataset: Any = MISSING
    
    # ================ Data Loading ================
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    
    # ================ Progress Tracking ================
    progress_bar: ProgressBarConfig = field(default_factory=ProgressBarConfig)
    
    # ================ Logging ================
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # ================ Checkpointing ================
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # ================ Reproducibility ================
    determinism: DeterminismConfig = field(default_factory=DeterminismConfig)
    
    # ================ Performance Optimization ================
    torch_compile: TorchCompileConfig = field(default_factory=TorchCompileConfig)
    
    # ================ Optimizer and Scheduler ================
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: Optional[LRSchedulerConfig] = None
    
    # ================ Optional Metrics ================
    metricAggregator: Optional[MetricAggregatorConfig] = None


@dataclass
class ExperimentConfig:
    """
    Master configuration for experiments that orchestrates training across multiple seeds.
    This is the top-level configuration that includes all training parameters.
    """
    # ================ Experiment Management ================
    experiment_id: Optional[str] = None  # Auto-generated if None
    seeds: List[int] = field(default_factory=lambda: [42])  # Default to single seed
    results_dir: str = "experiments"  # Base directory for experiment results (auto-created if doesn't exist)
    resume: bool = True  # Whether to resume interrupted experiments
    
    # ================ Core Training Configuration ================
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
