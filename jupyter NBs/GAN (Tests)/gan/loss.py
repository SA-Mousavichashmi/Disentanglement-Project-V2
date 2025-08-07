# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseLoss(ABC):
    """Base class for GAN losses."""
    
    def __init__(self, device='cuda'):
        """
        Initialize the base loss.
        
        Parameters
        ----------
        device : str
            Device to run computations on.
        """
        self.device = device
    
    @abstractmethod
    def discriminator_loss(self, real_output, fake_output, **kwargs):
        """
        Compute discriminator loss.
        
        Parameters
        ----------
        real_output : torch.Tensor
            Discriminator output for real images.
        fake_output : torch.Tensor
            Discriminator output for fake images.
            
        Returns
        -------
        torch.Tensor
            Discriminator loss.
        """
        pass
    
    @abstractmethod
    def generator_loss(self, fake_output, **kwargs):
        """
        Compute generator loss.
        
        Parameters
        ----------
        fake_output : torch.Tensor
            Discriminator output for fake images.
            
        Returns
        -------
        torch.Tensor
            Generator loss.
        """
        pass


class VanillaGANLoss(BaseLoss):
    """Vanilla GAN loss using binary cross-entropy."""
    
    def __init__(self, device='cuda', label_smoothing=0.0):
        """
        Initialize Vanilla GAN loss.
        
        Parameters
        ----------
        device : str
            Device to run computations on.
        label_smoothing : float
            Label smoothing factor (0.0 means no smoothing).
        """
        super().__init__(device)
        self.label_smoothing = label_smoothing
        self.criterion = nn.BCEWithLogitsLoss()
    
    def discriminator_loss(self, real_output, fake_output, **kwargs):
        """
        Compute vanilla GAN discriminator loss.
        
        The discriminator tries to classify real images as 1 and fake images as 0.
        """
        batch_size = real_output.size(0)
        
        # Real labels (with optional label smoothing)
        real_labels = torch.ones(batch_size, 1, device=self.device) - self.label_smoothing
        fake_labels = torch.zeros(batch_size, 1, device=self.device) + self.label_smoothing
        
        # Losses
        real_loss = self.criterion(real_output, real_labels)
        fake_loss = self.criterion(fake_output, fake_labels)
        
        return real_loss + fake_loss
    
    def generator_loss(self, fake_output, **kwargs):
        """
        Compute vanilla GAN generator loss.
        
        The generator tries to fool the discriminator by making fake images look real.
        """
        batch_size = fake_output.size(0)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        
        return self.criterion(fake_output, real_labels)


class LPGANLoss(BaseLoss):
    """LP-GAN loss with Lp penalty on gradients."""
    
    def __init__(self, device='cuda', lambda_gp=10.0, p=2):
        """
        Initialize LP-GAN loss.
        
        Parameters
        ----------
        device : str
            Device to run computations on.
        lambda_gp : float
            Gradient penalty coefficient.
        p : int
            Order of the Lp norm (typically 2 for L2 penalty).
        """
        super().__init__(device)
        self.lambda_gp = lambda_gp
        self.p = p
    
    def gradient_penalty(self, discriminator, real_samples, fake_samples):
        """
        Compute gradient penalty for LP-GAN.
        
        Parameters
        ----------
        discriminator : torch.nn.Module
            Discriminator network.
        real_samples : torch.Tensor
            Real images.
        fake_samples : torch.Tensor
            Generated images.
            
        Returns
        -------
        torch.Tensor
            Gradient penalty.
        """
        batch_size = real_samples.size(0)
        
        # Random interpolation between real and fake samples
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Discriminator output for interpolated samples
        disc_interpolated = discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Reshape gradients
        gradients = gradients.view(batch_size, -1)
        
        # Compute Lp norm of gradients
        gradient_norm = torch.norm(gradients, p=self.p, dim=1)
        
        # Gradient penalty: (||∇D(x)||_p - 1)^2
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return gradient_penalty
    
    def discriminator_loss(self, real_output, fake_output, discriminator=None, real_samples=None, fake_samples=None, **kwargs):
        """
        Compute LP-GAN discriminator loss.
        
        Uses Wasserstein loss with gradient penalty.
        """
        # Wasserstein loss: maximize E[D(x)] - E[D(G(z))]
        # For minimization: minimize E[D(G(z))] - E[D(x)]
        wasserstein_loss = torch.mean(fake_output) - torch.mean(real_output)
        
        # Gradient penalty
        if discriminator is not None and real_samples is not None and fake_samples is not None:
            gp = self.gradient_penalty(discriminator, real_samples, fake_samples)
            return wasserstein_loss + self.lambda_gp * gp
        else:
            return wasserstein_loss
    
    def generator_loss(self, fake_output, **kwargs):
        """
        Compute LP-GAN generator loss.
        
        Generator tries to maximize E[D(G(z))], so we minimize -E[D(G(z))].
        """
        return -torch.mean(fake_output)


class SNGANLoss(BaseLoss):
    """SN-GAN loss using hinge loss (typically used with spectral normalization)."""
    
    def __init__(self, device='cuda'):
        """
        Initialize SN-GAN loss.
        
        Parameters
        ----------
        device : str
            Device to run computations on.
        """
        super().__init__(device)
    
    def discriminator_loss(self, real_output, fake_output, **kwargs):
        """
        Compute SN-GAN discriminator loss using hinge loss.
        
        Discriminator loss: max(0, 1 - D(x)) + max(0, 1 + D(G(z)))
        """
        real_loss = F.relu(1.0 - real_output).mean()
        fake_loss = F.relu(1.0 + fake_output).mean()
        
        return real_loss + fake_loss
    
    def generator_loss(self, fake_output, **kwargs):
        """
        Compute SN-GAN generator loss.
        
        Generator loss: -E[D(G(z))]
        """
        return -fake_output.mean()


class WGANGPLoss(BaseLoss):
    """Wasserstein GAN loss with Gradient Penalty (WGAN-GP)."""
    
    def __init__(self, device='cuda', lambda_gp=10.0):
        """
        Initialize WGAN-GP loss.
        
        Parameters
        ----------
        device : str
            Device to run computations on.
        lambda_gp : float
            Gradient penalty coefficient.
        """
        super().__init__(device)
        self.lambda_gp = lambda_gp
    
    def gradient_penalty(self, discriminator, real_samples, fake_samples):
        """
        Compute gradient penalty for WGAN-GP.
        
        Parameters
        ----------
        discriminator : torch.nn.Module
            Discriminator network.
        real_samples : torch.Tensor
            Real images.
        fake_samples : torch.Tensor
            Generated images.
            
        Returns
        -------
        torch.Tensor
            Gradient penalty.
        """
        # Use the utility function but without lambda_gp multiplication
        # since we handle that in discriminator_loss
        return compute_gradient_penalty(
            discriminator, real_samples, fake_samples, 
            device=self.device, lambda_gp=1.0
        )
    
    def discriminator_loss(self, real_output, fake_output, discriminator=None, real_samples=None, fake_samples=None, **kwargs):
        """
        Compute WGAN-GP discriminator loss.
        
        Discriminator (critic) loss: E[D(G(z))] - E[D(x)] + λ * GP
        """
        # Wasserstein loss: E[D(G(z))] - E[D(x)]
        wasserstein_loss = torch.mean(fake_output) - torch.mean(real_output)
        
        # Gradient penalty
        if discriminator is not None and real_samples is not None and fake_samples is not None:
            gp = self.gradient_penalty(discriminator, real_samples, fake_samples)
            return wasserstein_loss + self.lambda_gp * gp
        else:
            return wasserstein_loss
    
    def generator_loss(self, fake_output, **kwargs):
        """
        Compute WGAN-GP generator loss.
        
        Generator loss: -E[D(G(z))]
        """
        return -torch.mean(fake_output)


class LSGANLoss(BaseLoss):
    """Least Squares GAN loss using mean squared error."""
    
    def __init__(self, device='cuda', a=0.0, b=1.0, c=1.0):
        """
        Initialize LSGAN loss.
        
        Parameters
        ----------
        device : str
            Device to run computations on.
        a : float
            Target value for fake data in discriminator loss (typically 0).
        b : float
            Target value for fake data in generator loss (typically 1).
        c : float
            Target value for real data in discriminator loss (typically 1).
        """
        super().__init__(device)
        self.a = a  # fake label for discriminator
        self.b = b  # fake label for generator (what generator wants discriminator to output)
        self.c = c  # real label for discriminator
        self.criterion = nn.MSELoss()
    
    def discriminator_loss(self, real_output, fake_output, **kwargs):
        """
        Compute LSGAN discriminator loss.
        
        Discriminator loss: 0.5 * [(D(x) - c)^2 + (D(G(z)) - a)^2]
        where c is the target for real data and a is the target for fake data.
        """
        batch_size = real_output.size(0)
        
        # Target labels
        real_targets = torch.full((batch_size, 1), self.c, device=self.device)
        fake_targets = torch.full((batch_size, 1), self.a, device=self.device)
        
        # Losses
        real_loss = self.criterion(real_output, real_targets)
        fake_loss = self.criterion(fake_output, fake_targets)
        
        return 0.5 * (real_loss + fake_loss)
    
    def generator_loss(self, fake_output, **kwargs):
        """
        Compute LSGAN generator loss.
        
        Generator loss: 0.5 * (D(G(z)) - b)^2
        where b is what the generator wants the discriminator to output for fake data.
        """
        batch_size = fake_output.size(0)
        target_labels = torch.full((batch_size, 1), self.b, device=self.device)
        
        return 0.5 * self.criterion(fake_output, target_labels)


def get_loss(loss_type, **kwargs):
    """
    Factory function to get loss based on type.
    
    Parameters
    ----------
    loss_type : str
        Type of loss ('vanilla', 'lpgan', 'sngan', 'wgan').
    **kwargs
        Additional arguments for loss initialization.
        
    Returns
    -------
    BaseLoss
        Loss function instance.
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'vanilla':
        return VanillaGANLoss(**kwargs)
    elif loss_type == 'lpgan':
        return LPGANLoss(**kwargs)
    elif loss_type == 'sngan':
        return SNGANLoss(**kwargs)
    elif loss_type == 'wgan':
        return WGANGPLoss(**kwargs)
    elif loss_type == 'lsgan':
        return LSGANLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Available types: ['vanilla', 'lpgan', 'sngan', 'wgan', 'lsgan']")


# Utility functions for loss computation
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device='cuda', lambda_gp=10.0):
    """
    Standalone function to compute gradient penalty for WGAN-GP style losses.
    
    Parameters
    ----------
    discriminator : torch.nn.Module
        Discriminator network.
    real_samples : torch.Tensor
        Real images.
    fake_samples : torch.Tensor
        Generated images.
    device : str
        Device to run computations on.
    lambda_gp : float
        Gradient penalty coefficient.
        
    Returns
    -------
    torch.Tensor
        Gradient penalty.
    """
    batch_size = real_samples.size(0)
    
    # Random interpolation between real and fake samples
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Discriminator output for interpolated samples
    disc_interpolated = discriminator(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=disc_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(disc_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Reshape gradients
    gradients = gradients.view(batch_size, -1)
    
    # Compute L2 norm of gradients
    gradient_norm = torch.norm(gradients, p=2, dim=1)
    
    # Gradient penalty: (||∇D(x)||_2 - 1)^2
    gradient_penalty = lambda_gp * torch.mean((gradient_norm - 1) ** 2)
    
    return gradient_penalty
