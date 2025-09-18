# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
GAN implementation with multiple architectures and loss functions.

This package provides:
- Generator and Discriminator architectures based on Locatello VAE components
- Multiple loss functions: Vanilla GAN, LP-GAN, SN-GAN, WGAN-GP
- Complete training framework with checkpointing and visualization
"""

from .architecture import Generator, Discriminator, create_gan
from .loss import (
    VanillaGANLoss, 
    LPGANLoss, 
    SNGANLoss, 
    WGANGPLoss, 
    get_loss,
    compute_gradient_penalty
)
from .trainer import GANTrainer, create_trainer

__all__ = [
    'Generator',
    'Discriminator', 
    'create_gan',
    'VanillaGANLoss',
    'LPGANLoss',
    'SNGANLoss', 
    'WGANGPLoss',
    'get_loss',
    'compute_gradient_penalty',
    'GANTrainer',
    'create_trainer'
]
