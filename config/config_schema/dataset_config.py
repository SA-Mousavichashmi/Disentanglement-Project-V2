"""
Configuration schema for all datasets used in the codebase.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import MISSING # type: ignore


@dataclass
class DatasetConfig:
    """Base class for dataset configuration."""
    name: str = MISSING
    selected_factors: Any = 'all'  # Default is 'all', can also be list of factor names
    not_selected_factors_index_value: Optional[Dict[str, Any]] = None
    root: str = MISSING
    subset: float = 1.0
    transforms: Optional[List[str]] = None


@dataclass
class Cars3DConfig(DatasetConfig):
    """Configuration for Cars3D dataset."""
    name: str = "cars3d"
    not_selected_factors_index_value: Dict[str, Any] = field(default_factory=dict)
    root: str = "data/cars3d/"


@dataclass
class DSpritesConfig(DatasetConfig):
    """Configuration for DSprites dataset."""
    name: str = "dsprites"
    not_selected_factors_index_value: Dict[str, Any] = field(default_factory=dict)
    root: str = "data/dsprites/"
    drop_color_factor: bool = True


@dataclass
class CelebAConfig(DatasetConfig):
    """Configuration for CelebA dataset."""
    name: str = "celeba"
    root: str = "data/celeba/"
    crop_faces: bool = False
    crop_margin: float = 0.6
    resize_algorithm: str = "LANCZOS"
    force_download: bool = False


@dataclass
class Shapes3DConfig(DatasetConfig):
    """Configuration for Shapes3D dataset."""
    name: str = "shapes3d"
    not_selected_factors_index_value: Dict[str, Any] = field(default_factory=dict)
    root: str = "data/shapes3d/"

