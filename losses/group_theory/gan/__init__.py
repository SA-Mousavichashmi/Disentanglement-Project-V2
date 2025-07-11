"""
GAN module for Group Theory Implementation.
"""

from .trainer import GANTrainer
from .losses import select_gan_loss
from .discrim_archs import select_discriminator

__all__ = ['GANTrainer', 'select_gan_loss', 'select_discriminator']
