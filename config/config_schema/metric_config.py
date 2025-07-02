"""
Configuration schema for all metrics used in the codebase.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union
from omegaconf import MISSING # type: ignore


@dataclass
class BaseMetricConfig:
    """Base class for metric configuration."""
    name: str = MISSING

@dataclass
class MIGConfig(BaseMetricConfig):
    """Configuration for Mutual Information Gap metric."""
    name: str = "mig"
    num_bins: int = 20
    num_workers: int = 8
    mi_method: str = 'pyitlib'
    entropy_method: str = 'pyitlib'

@dataclass
class SAPdConfig(BaseMetricConfig):
    """Configuration for Separated Attribute Predictability metric."""
    name: str = "sap_d"
    num_train: int = 10000
    num_test: int = 5000
    num_bins: int = 20

@dataclass
class DCIdConfig(BaseMetricConfig):
    """Configuration for Disentanglement, Completeness, and Informativeness metric."""
    name: str = "dci_d"
    num_train: Optional[int] = 10000
    num_test: Optional[int] = 5000
    split_ratio: Optional[float] = None
    backend: str = 'sklearn'
    num_workers: int = 8

@dataclass
class ModularitydConfig(BaseMetricConfig):
    """Configuration for Modularity metric."""
    name: str = "modularity_d"
    num_bins: int = 20
    num_workers: int = 8
    mi_method: str = 'sklearn'

@dataclass
class ReconstructionErrorConfig(BaseMetricConfig):
    """Configuration for Reconstruction Error metric."""
    name: str = "reconstruction_error"
    error_type: str = "mse"  # Options: "mse" or "ce"

@dataclass
class MetricAggregatorConfig:
    """Configuration for the Metric Aggregator."""
    metrics: List[BaseMetricConfig] = field(default_factory=list)
    sample_num: Optional[int] = None  # Number of samples to use for metric computation