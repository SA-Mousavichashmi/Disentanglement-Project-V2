"""
Group Theory Losses Module

This module provides base and specialized implementations for group theory losses
that can be applied to VAE models with different latent space topologies.

Available Classes:
- BaseGroupTheoryLoss: Abstract base class with common functionality
- GroupTheoryNVAELoss: Implementation for R^1 topology (translation-only) 
- GroupTheorySNVAELoss: Implementation for mixed R^1 and S^1 topologies
"""

from .base_group_theory import BaseGroupTheoryLoss
from .n_vae.group_theory import GroupTheoryNVAELoss
from .s_n_vae.group_theory import GroupTheorySNVAELoss

__all__ = ['BaseGroupTheoryLoss', 'GroupTheoryNVAELoss', 'GroupTheorySNVAELoss']
