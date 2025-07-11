"""
Refactored GAN Trainer for Group Theory Implementation.
This module provides a flexible, extensible GAN training system with registry-based
architecture and loss selection.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union, List, Tuple
from .losses import select_gan_loss
from .discrim_archs import select_discriminator


class GANTrainer:
    """
    Unified GAN trainer that handles different architectures and losses.
    
    This class provides a flexible interface for training GANs with various
    architectures and loss functions. It supports:
    - Multiple loss functions (WGAN-GP, Hinge, BCE, LSGAN, etc.)
    - Multiple architectures (Locatello, Spectral Norm, etc.)
    """
    
    def __init__(self, 
                 img_size: Tuple[int, int, int],  # (channels, height, width)
                 device: torch.device,
                 lr: float = 1e-4,
                 loss_type: str = "wgan_gp",
                 architecture_type: str = "locatello",
                 loss_kwargs: Optional[Dict] = None,
                 architecture_kwargs: Optional[Dict] = None,
                 optimizer_type: str = "adam",
                 optimizer_kwargs: Optional[Dict] = None):
        """
        Initialize GAN trainer.
        
        Args:
            img_size: Tuple of (channels, height, width) for input images
            device: Device to place the discriminator on
            lr: Learning rate for the discriminator optimizer
            loss_type: Type of GAN loss (e.g., "wgan_gp", "hinge", "bce")
            architecture_type: Type of discriminator architecture (e.g., "locatello", "spectral_norm")
            loss_kwargs: Additional kwargs for loss function
            architecture_kwargs: Additional kwargs for architecture
            optimizer_type: Type of optimizer for discriminator
            optimizer_kwargs: Additional kwargs for optimizer
        """
        self.img_size = img_size
        self.device = device
        self.lr = lr
        self.loss_type = loss_type
        self.architecture_type = architecture_type
        self.loss_kwargs = loss_kwargs or {}
        self.architecture_kwargs = architecture_kwargs or {}
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs or {}
        
        # Initialize loss function
        self.loss_fn = select_gan_loss(loss_type, **self.loss_kwargs)
        
        # Initialize discriminator and optimizer
        self._initialize_discriminator()
        
    def _initialize_discriminator(self):
        """
        Private method to initialize discriminator and optimizer.
        """
        input_channels = self.img_size[0]
        
        # Create discriminator
        self.discriminator = select_discriminator(
            architecture_type=self.architecture_type,
            input_channels=input_channels,
            **self.architecture_kwargs
        ).to(self.device)
        
        # Create optimizer
        optimizer_kwargs = {**self.optimizer_kwargs, 'lr': self.lr}
        if self.optimizer_type.lower() == "adam":
            self.discriminator_optimizer = torch.optim.Adam(
                self.discriminator.parameters(), 
                **optimizer_kwargs
            )
        elif self.optimizer_type.lower() == "rmsprop":
            self.discriminator_optimizer = torch.optim.RMSprop(
                self.discriminator.parameters(), 
                **optimizer_kwargs
            )
        elif self.optimizer_type.lower() == "sgd":
            self.discriminator_optimizer = torch.optim.SGD(
                self.discriminator.parameters(), 
                **optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
    
    def train_discriminator(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        Train discriminator for one step (single-scale only).
        
        Args:
            real_images: Real images tensor
            fake_images: Fake/generated images tensor
        
        Returns:
            float: Discriminator loss value
        """
        self.discriminator_optimizer.zero_grad()
        real_output = self.discriminator(real_images)
        fake_output = self.discriminator(fake_images.detach())
        d_loss = self.loss_fn.discriminator_loss(
            real_output=real_output,
            fake_output=fake_output,
            real_images=real_images,
            fake_images=fake_images,
            discriminator=self.discriminator
        )
        d_loss.backward()
        self.discriminator_optimizer.step()
        return d_loss.item()
    
    def compute_generator_loss(self, fake_images: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss (single-scale only).
        
        Args:
            fake_images: Generated/fake images tensor
        
        Returns:
            torch.Tensor: Generator loss
        """
        fake_output = self.discriminator(fake_images)
        return self.loss_fn.generator_loss(fake_output)
    
    def set_training_mode(self, mode: bool = True):
        """Set training mode for discriminator."""
        self.discriminator.train(mode)
    
    def freeze_discriminator(self):
        """Freeze discriminator parameters."""
        for param in self.discriminator.parameters():
            param.requires_grad_(False)
    
    def unfreeze_discriminator(self):
        """Unfreeze discriminator parameters."""
        for param in self.discriminator.parameters():
            param.requires_grad_(True)
    
    def get_discriminator_output(self, images: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Get discriminator output without training.
        
        Args:
            images: Input images
            
        Returns:
            Discriminator output(s)
        """
        with torch.no_grad():
            return self.discriminator(images)
    
    def state_dict(self) -> Dict:
        """Get state dict for saving."""
        state = {
            'img_size': self.img_size,
            'lr': self.lr,
            'loss_type': self.loss_type,
            'architecture_type': self.architecture_type,
            'loss_kwargs': self.loss_kwargs,
            'architecture_kwargs': self.architecture_kwargs,
            'optimizer_type': self.optimizer_type,
            'optimizer_kwargs': self.optimizer_kwargs,
            'discriminator_state_dict': self.discriminator.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
        }
        return state
    
    def load_state_dict(self, state_dict: Dict):
        """Load state dict."""
        # Update configuration
        self.img_size = state_dict.get('img_size', self.img_size)
        self.lr = state_dict.get('lr', self.lr)
        self.loss_type = state_dict.get('loss_type', self.loss_type)
        self.architecture_type = state_dict.get('architecture_type', self.architecture_type)
        self.loss_kwargs = state_dict.get('loss_kwargs', self.loss_kwargs)
        self.architecture_kwargs = state_dict.get('architecture_kwargs', self.architecture_kwargs)
        self.optimizer_type = state_dict.get('optimizer_type', self.optimizer_type)
        self.optimizer_kwargs = state_dict.get('optimizer_kwargs', self.optimizer_kwargs)
        
        # Recreate loss function
        self.loss_fn = select_gan_loss(self.loss_type, **self.loss_kwargs)
        
        # Recreate discriminator and optimizer
        self._initialize_discriminator()
        
        # Load states
        if state_dict.get('discriminator_state_dict') is not None:
            self.discriminator.load_state_dict(state_dict['discriminator_state_dict'])
        if state_dict.get('discriminator_optimizer_state_dict') is not None:
            self.discriminator_optimizer.load_state_dict(state_dict['discriminator_optimizer_state_dict'])
        
    def __repr__(self) -> str:
        return (f"GANTrainer(img_size={self.img_size}, loss_type='{self.loss_type}', "
                f"architecture_type='{self.architecture_type}', device={self.device})")
