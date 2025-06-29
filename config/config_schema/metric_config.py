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
    num_train: Optional[int] = None
    num_test: Optional[int] = None
    split_ratio: Optional[float] = 0.8
    backend: str = 'sklearn'
    num_workers: int = 8

@dataclass
class KLDConfig(BaseMetricConfig):
    """Configuration for KL-Divergence metric."""
    name: str = "kld"
    batch_size: int = MISSING

@dataclass
class RandKLDConfig(BaseMetricConfig):
    """Configuration for random KL-Divergence metric."""
    name: str = "rand_kld"
    batch_size: int = MISSING

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

MetricConfigType = Union[
    MIGConfig,
    SAPdConfig,
    DCIdConfig,
    KLDConfig,
    RandKLDConfig,
    ModularitydConfig,
    ReconstructionErrorConfig,
]

@dataclass
class MetricAggregatorConfig(BaseMetricConfig):
    """Configuration for the Metric Aggregator."""
    name: str = "metric_aggregator"
    metrics: List[MetricConfigType] = field(default_factory=list)
    sample_num: int = 10000  # Number of samples to use for metric computation