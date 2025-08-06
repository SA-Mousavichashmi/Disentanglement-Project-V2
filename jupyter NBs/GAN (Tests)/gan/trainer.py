# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from pathlib import Path

from .architecture import Generator, Discriminator
from .loss import get_loss


class GANTrainer:
    """Trainer for GAN models supporting various loss functions and architectures."""
    
    def __init__(self, 
                 generator=None,
                 discriminator=None,
                 loss_type='vanilla',
                 loss_kwargs=None,
                 g_optimizer=None,
                 d_optimizer=None,
                 g_lr=2e-4,
                 d_lr=2e-4,
                 beta1=0.5,
                 beta2=0.999,
                 device='cuda',
                 latent_dim=10,
                 img_size=(3, 64, 64),
                 n_critic=1,
                 clip_value=None):
        """
        Initialize GAN trainer.
        
        Parameters
        ----------
        generator : torch.nn.Module, optional
            Generator network. If None, will create default generator.
        discriminator : torch.nn.Module, optional  
            Discriminator network. If None, will create default discriminator.
        loss_type : str
            Type of loss function ('vanilla', 'lpgan', 'sngan', 'wgan').
        loss_kwargs : dict, optional
            Additional arguments for loss function.
        g_optimizer : torch.optim.Optimizer, optional
            Generator optimizer. If None, will create Adam optimizer.
        d_optimizer : torch.optim.Optimizer, optional
            Discriminator optimizer. If None, will create Adam optimizer.
        g_lr : float
            Generator learning rate.
        d_lr : float
            Discriminator learning rate.
        beta1 : float
            Beta1 parameter for Adam optimizer.
        beta2 : float
            Beta2 parameter for Adam optimizer.
        device : str
            Device to train on.
        latent_dim : int
            Dimensionality of latent space.
        img_size : tuple
            Size of images.
        n_critic : int
            Number of discriminator updates per generator update.
        clip_value : float, optional
            Value to clip discriminator weights (for WGAN).
        """
        self.device = device
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.n_critic = n_critic
        self.clip_value = clip_value
        self.loss_type = loss_type
        
        # Initialize networks
        if generator is None:
            self.generator = Generator(latent_dim=latent_dim, img_size=img_size)
        else:
            self.generator = generator
        
        if discriminator is None:
            # Use spectral norm for SN-GAN
            use_spectral_norm = (loss_type.lower() == 'sngan')
            self.discriminator = Discriminator(img_size=img_size, use_spectral_norm=use_spectral_norm)
        else:
            self.discriminator = discriminator
        
        # Move to device
        self.generator.to(device)
        self.discriminator.to(device)
        
        # Initialize loss function
        if loss_kwargs is None:
            loss_kwargs = {}
        loss_kwargs['device'] = device
        self.loss_fn = get_loss(loss_type, **loss_kwargs)
        
        # Initialize optimizers
        if g_optimizer is None:
            self.g_optimizer = optim.Adam(self.generator.parameters(), lr=g_lr, betas=(beta1, beta2))
        else:
            self.g_optimizer = g_optimizer
            
        if d_optimizer is None:
            self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=(beta1, beta2))
        else:
            self.d_optimizer = d_optimizer
        
        # Training history
        self.history = defaultdict(list)
        self.current_epoch = 0
    
    def generate_noise(self, batch_size):
        """Generate random noise for generator input."""
        return torch.randn(batch_size, self.latent_dim, device=self.device)
    
    def train_discriminator(self, real_images):
        """Train discriminator for one step."""
        batch_size = real_images.size(0)
        
        # Reset gradients
        self.d_optimizer.zero_grad()
        
        # Generate fake images
        noise = self.generate_noise(batch_size)
        with torch.no_grad():
            fake_images = self.generator(noise)
        
        # Get discriminator outputs
        real_output = self.discriminator(real_images)
        fake_output = self.discriminator(fake_images.detach())
        
        # Compute loss
        if self.loss_type.lower() == 'lpgan':
            d_loss = self.loss_fn.discriminator_loss(
                real_output, fake_output,
                discriminator=self.discriminator,
                real_samples=real_images,
                fake_samples=fake_images
            )
        else:
            d_loss = self.loss_fn.discriminator_loss(real_output, fake_output)
        
        # Backward pass
        d_loss.backward()
        self.d_optimizer.step()
        
        # Clip weights for WGAN
        if self.clip_value is not None:
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)
        
        return d_loss.item()
    
    def train_generator(self, batch_size):
        """Train generator for one step."""
        # Reset gradients
        self.g_optimizer.zero_grad()
        
        # Generate fake images
        noise = self.generate_noise(batch_size)
        fake_images = self.generator(noise)
        
        # Get discriminator output for fake images
        fake_output = self.discriminator(fake_images)
        
        # Compute loss
        g_loss = self.loss_fn.generator_loss(fake_output)
        
        # Backward pass
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        d_losses = []
        g_losses = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for i, (real_images, _) in enumerate(pbar):
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            
            # Train discriminator
            d_loss = self.train_discriminator(real_images)
            d_losses.append(d_loss)
            
            # Train generator (every n_critic steps)
            if i % self.n_critic == 0:
                g_loss = self.train_generator(batch_size)
                g_losses.append(g_loss)
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': f'{np.mean(d_losses[-10:]):.4f}',
                'G_loss': f'{np.mean(g_losses[-10:]):.4f}' if g_losses else 'N/A'
            })
        
        # Record epoch losses
        self.history['d_loss'].append(np.mean(d_losses))
        if g_losses:
            self.history['g_loss'].append(np.mean(g_losses))
        
        return np.mean(d_losses), np.mean(g_losses) if g_losses else 0.0
    
    def train(self, dataloader, epochs):
        """
        Train the GAN.
        
        Parameters
        ----------
        dataloader : DataLoader
            Training data loader.
        epochs : int
            Number of epochs to train.
        save_dir : str, optional
            Directory to save checkpoints and samples.
        sample_interval : int
            Interval to generate and save sample images.
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Loss type: {self.loss_type}")
        print(f"Device: {self.device}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            d_loss, g_loss = self.train_epoch(dataloader, epoch)
            
            print(f"Epoch [{epoch+1}/{epochs}] - D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")
        
        print("Training completed!")
    
    def plot_losses(self, save_path=None):
        """Plot training losses."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'd_loss' in self.history:
            ax.plot(self.history['d_loss'], label='Discriminator Loss', alpha=0.7)
        if 'g_loss' in self.history:
            ax.plot(self.history['g_loss'], label='Generator Loss', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'GAN Training Losses ({self.loss_type.upper()})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def generate_samples(self, n_samples=16):
        """Generate sample images."""
        self.generator.eval()
        
        with torch.no_grad():
            noise = self.generate_noise(n_samples)
            fake_images = self.generator(noise)
            
            # Denormalize images from [-1, 1] to [0, 1]
            fake_images = (fake_images + 1) / 2
            fake_images = torch.clamp(fake_images, 0, 1)
        
        self.generator.train()
        return fake_images.cpu()

    def plot_samples(self, n_samples=16, n_cols=4, figsize=(8, 8), save_path=None):
        """Plot generated sample images in a grid."""
        samples = self.generate_samples(n_samples).cpu()
        n_rows = int(np.ceil(n_samples / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)
        for i in range(n_samples):
            img = samples[i].permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].axis('off')
        # Hide unused axes
        for j in range(n_samples, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

def create_trainer(generator_config=None, discriminator_config=None, trainer_config=None):
    """
    Factory function to create a GAN trainer with specified configurations.
    
    Parameters
    ----------
    generator_config : dict, optional
        Configuration for generator.
    discriminator_config : dict, optional
        Configuration for discriminator.
    trainer_config : dict, optional
        Configuration for trainer.
        
    Returns
    -------
    GANTrainer
        Configured GAN trainer.
    """
    if generator_config is None:
        generator_config = {}
    if discriminator_config is None:
        discriminator_config = {}
    if trainer_config is None:
        trainer_config = {}
    
    # Create networks
    generator = Generator(**generator_config)
    discriminator = Discriminator(**discriminator_config)
    
    # Create trainer
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        **trainer_config
    )
    
    return trainer