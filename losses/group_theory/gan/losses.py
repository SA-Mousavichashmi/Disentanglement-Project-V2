"""
GAN Loss Functions for Group Theory Implementation.
This module provides a registry-based system for different GAN loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


class BaseGANLoss(ABC):
    """Base class for GAN loss implementations."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._name = None
    
    @abstractmethod
    def discriminator_loss(self, real_output: torch.Tensor, fake_output: torch.Tensor, 
                          real_images: torch.Tensor = None, fake_images: torch.Tensor = None,
                          discriminator: nn.Module = None) -> torch.Tensor:
        """Calculate discriminator loss."""
        pass
    
    @abstractmethod
    def generator_loss(self, fake_output: torch.Tensor) -> torch.Tensor:
        """Calculate generator loss."""
        pass
    
    @property
    def name(self) -> str:
        if self._name is None:
            return self.__class__.__name__.replace('Loss', '').lower()
        return self._name
    
    def set_name(self, name: str):
        """Allow custom naming for loss instances."""
        self._name = name


class WGANGPLoss(BaseGANLoss):
    """Wasserstein GAN with Gradient Penalty loss."""
    
    def __init__(self, gradient_penalty_weight: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.gradient_penalty_weight = gradient_penalty_weight
    
    def _compute_gradient_penalty(self, discriminator: nn.Module, 
                                 real_images: torch.Tensor, 
                                 fake_images: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        device = real_images.device
        batch_size = real_images.size(0)
        
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        alpha = alpha.expand_as(real_images)
        
        interpolates = alpha * real_images + (1 - alpha) * fake_images
        interpolates.requires_grad_(True)
        
        disc_interpolates = discriminator(interpolates)
        
        grad_outputs = torch.ones_like(disc_interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def discriminator_loss(self, real_output: torch.Tensor, fake_output: torch.Tensor,
                          real_images: torch.Tensor = None, fake_images: torch.Tensor = None,
                          discriminator: nn.Module = None) -> torch.Tensor:
        """WGAN-GP discriminator loss: -E[D(real)] + E[D(fake)] + λ*GP"""
        if real_images is None or fake_images is None or discriminator is None:
            raise ValueError("WGAN-GP requires real_images, fake_images, and discriminator for gradient penalty")
        
        gp = self._compute_gradient_penalty(discriminator, real_images, fake_images)
        d_loss = -real_output.mean() + fake_output.mean() + self.gradient_penalty_weight * gp
        return d_loss
    
    def generator_loss(self, fake_output: torch.Tensor) -> torch.Tensor:
        """WGAN-GP generator loss: -E[D(fake)]"""
        return -fake_output.mean()


class HingeLoss(BaseGANLoss):
    """Hinge loss for SN-GAN and other spectral normalization approaches."""
    
    def discriminator_loss(self, real_output: torch.Tensor, fake_output: torch.Tensor,
                          real_images: torch.Tensor = None, fake_images: torch.Tensor = None,
                          discriminator: nn.Module = None) -> torch.Tensor:
        """Hinge loss discriminator: max(0, 1-D(real)) + max(0, 1+D(fake))"""
        d_loss_real = torch.relu(1.0 - real_output).mean()
        d_loss_fake = torch.relu(1.0 + fake_output).mean()
        return d_loss_real + d_loss_fake
    
    def generator_loss(self, fake_output: torch.Tensor) -> torch.Tensor:
        """Hinge loss generator: -E[D(fake)]"""
        return -fake_output.mean()


class BCELoss(BaseGANLoss):
    """Binary Cross-Entropy loss for vanilla GAN."""
    
    def __init__(self, label_smoothing: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.criterion = nn.BCEWithLogitsLoss()
    
    def discriminator_loss(self, real_output: torch.Tensor, fake_output: torch.Tensor,
                          real_images: torch.Tensor = None, fake_images: torch.Tensor = None,
                          discriminator: nn.Module = None) -> torch.Tensor:
        """BCE discriminator loss with optional label smoothing."""
        batch_size = real_output.size(0)
        device = real_output.device
        
        # Apply label smoothing if specified
        real_label_value = 1.0 - self.label_smoothing
        fake_label_value = 0.0 + self.label_smoothing
        
        real_labels = torch.full((batch_size, 1), real_label_value, device=device)
        fake_labels = torch.full((batch_size, 1), fake_label_value, device=device)
        
        d_loss_real = self.criterion(real_output, real_labels)
        d_loss_fake = self.criterion(fake_output, fake_labels)
        return d_loss_real + d_loss_fake
    
    def generator_loss(self, fake_output: torch.Tensor) -> torch.Tensor:
        """BCE generator loss."""
        batch_size = fake_output.size(0)
        device = fake_output.device
        
        real_labels = torch.ones(batch_size, 1, device=device)
        return self.criterion(fake_output, real_labels)


class LSGANLoss(BaseGANLoss):
    """Least Squares GAN loss."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterion = nn.MSELoss()
    
    def discriminator_loss(self, real_output: torch.Tensor, fake_output: torch.Tensor,
                          real_images: torch.Tensor = None, fake_images: torch.Tensor = None,
                          discriminator: nn.Module = None) -> torch.Tensor:
        """LSGAN discriminator loss."""
        batch_size = real_output.size(0)
        device = real_output.device
        
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        d_loss_real = self.criterion(real_output, real_labels)
        d_loss_fake = self.criterion(fake_output, fake_labels)
        return d_loss_real + d_loss_fake
    
    def generator_loss(self, fake_output: torch.Tensor) -> torch.Tensor:
        """LSGAN generator loss."""
        batch_size = fake_output.size(0)
        device = fake_output.device
        
        real_labels = torch.ones(batch_size, 1, device=device)
        return self.criterion(fake_output, real_labels)


class WGANLoss(BaseGANLoss):
    """Standard Wasserstein GAN loss (without gradient penalty)."""
    
    def discriminator_loss(self, real_output: torch.Tensor, fake_output: torch.Tensor,
                          real_images: torch.Tensor = None, fake_images: torch.Tensor = None,
                          discriminator: nn.Module = None) -> torch.Tensor:
        """WGAN discriminator loss: -E[D(real)] + E[D(fake)]"""
        return -real_output.mean() + fake_output.mean()
    
    def generator_loss(self, fake_output: torch.Tensor) -> torch.Tensor:
        """WGAN generator loss: -E[D(fake)]"""
        return -fake_output.mean()


_losses = {
    "wgan_gp": WGANGPLoss,
    "wgan": WGANLoss,
    "hinge": HingeLoss,
    "bce": BCELoss,
    "lsgan": LSGANLoss,
}

def select_gan_loss(name: str, **kwargs) -> BaseGANLoss:
    """
    Select and create a GAN loss function instance based on its name.
    
    Args:
        name (str): The name of the GAN loss function (e.g., "wgan_gp", "hinge", "bce").
        **kwargs: Arbitrary keyword arguments to pass to the loss function's constructor.
        
    Returns:
        BaseGANLoss: An instance of the selected GAN loss function.
        
    Raises:
        ValueError: If an unknown loss type is provided.
    """
    if name not in _losses:
        available = list(_losses.keys())
        raise ValueError(f"Unknown loss type: {name}. Available: {available}")
    
    loss_instance = _losses[name](**kwargs)
    loss_instance.set_name(name)
    return loss_instance
