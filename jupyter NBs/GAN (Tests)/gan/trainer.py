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
                 generator,
                 discriminator,
                 g_optimizer,
                 d_optimizer,
                 loss_type='vanilla',
                 loss_kwargs=None,
                 device='cuda',
                 n_critic=1,
                 clip_value=None,
                 progress_interval=100
                 ): 
        """
        Initialize GAN trainer.
        
        Parameters
        ----------
        generator : torch.nn.Module
            Generator network.
        discriminator : torch.nn.Module  
            Discriminator network.
        loss_type : str
            Type of loss function ('vanilla', 'lpgan', 'sngan', 'wgan').
        loss_kwargs : dict, optional
            Additional arguments for loss function.
        g_optimizer : torch.optim.Optimizer
            Generator optimizer (must be provided).
        d_optimizer : torch.optim.Optimizer
            Discriminator optimizer (must be provided).
        device : str
            Device to train on.
        n_critic : int
            Number of discriminator updates per generator update.
        clip_value : float, optional
            Value to clip discriminator weights (for WGAN).
        progress_interval : int
            Interval to update progress bar in train_epoch.
        """
        self.device = device
        self.n_critic = n_critic
        self.clip_value = clip_value
        self.loss_type = loss_type
        self.progress_interval = progress_interval  # <-- set instance variable
        
        # Initialize networks
        if generator is None or discriminator is None:
            raise ValueError("Generator and Discriminator must be provided. Predefined models are no longer used.")
        self.generator = generator
        self.discriminator = discriminator
        
        # Move to device
        self.generator.to(device)
        self.discriminator.to(device)
        
        # Initialize loss function
        if loss_kwargs is None:
            loss_kwargs = {}
        loss_kwargs['device'] = device
        self.loss_fn = get_loss(loss_type, **loss_kwargs)
        
        # Optimizers must be provided externally
        if g_optimizer is None or d_optimizer is None:
            raise ValueError("g_optimizer and d_optimizer must be provided externally.")
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        
        # Training history
        self.history = defaultdict(list)
        self.current_epoch = 0
        
        self.latent_dim = self.generator.latent_dim
    
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
            
            # Update progress bar only every self.progress_interval steps
            if i % self.progress_interval == 0 or i == len(dataloader) - 1:
                pbar.set_postfix({
                    'D_loss': f'{np.mean(d_losses[-10:]):.4f}',
                    'G_loss': f'{np.mean(g_losses[-10:]):.4f}' if g_losses else 'N/A'
                })
        
        # Record epoch losses
        self.history['d_loss'].append(np.mean(d_losses))
        if g_losses:
            self.history['g_loss'].append(np.mean(g_losses))
        
        return np.mean(d_losses), np.mean(g_losses) if g_losses else 0.0
    
    def train_iterations(self, dataloader, total_iterations, log_interval=None):
        """
        Train the GAN for a specified number of iterations.
        
        Parameters
        ----------
        dataloader : DataLoader
            Training data loader.
        total_iterations : int
            Total number of iterations to train.
        log_interval : int, optional
            Interval to log losses (default: len(dataloader), i.e., every epoch equivalent).
        """
        if log_interval is None:
            log_interval = len(dataloader)
            
        print(f"Starting training for {total_iterations} iterations...")
        print(f"Loss type: {self.loss_type}")
        print(f"Device: {self.device}")
        print(f"Log interval: {log_interval} iterations")
        
        self.generator.train()
        self.discriminator.train()
        
        # Create iterator over dataloader
        data_iter = iter(dataloader)
        
        # Track losses for logging
        d_losses = []
        g_losses = []
        
        # Track losses for history (logged every log_interval)
        d_losses_epoch = []
        g_losses_epoch = []
        
        pbar = tqdm(range(total_iterations), desc='Training Iterations')
        
        for iteration in pbar:
            # Get next batch, reset iterator if exhausted
            try:
                real_images, _ = next(data_iter)
            except StopIteration:
                # Reset the iterator when dataloader is exhausted
                data_iter = iter(dataloader)
                real_images, _ = next(data_iter)
                
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            
            # Update current epoch for compatibility
            self.current_epoch = iteration // len(dataloader)
            
            # Train discriminator
            d_loss = self.train_discriminator(real_images)
            d_losses.append(d_loss)
            d_losses_epoch.append(d_loss)
            
            # Train generator (every n_critic steps)
            if iteration % self.n_critic == 0:
                g_loss = self.train_generator(batch_size)
                g_losses.append(g_loss)
                g_losses_epoch.append(g_loss)
            
            # Update progress bar only every self.progress_interval steps
            if iteration % self.progress_interval == 0 or iteration == total_iterations - 1:
                pbar.set_postfix({
                    'D_loss': f'{np.mean(d_losses[-10:]):.4f}',
                    'G_loss': f'{np.mean(g_losses[-10:]):.4f}' if g_losses else 'N/A',
                    'Epoch': f'{iteration // len(dataloader):.1f}'
                })
            
            # Log losses every log_interval iterations
            if (iteration + 1) % log_interval == 0:
                # Record losses to history
                self.history['d_loss'].append(np.mean(d_losses_epoch))
                if g_losses_epoch:
                    self.history['g_loss'].append(np.mean(g_losses_epoch))
                
                # Reset epoch loss tracking
                d_losses_epoch = []
                g_losses_epoch = []
        
        # Record final losses if there are remaining losses
        if d_losses_epoch:
            self.history['d_loss'].append(np.mean(d_losses_epoch))
        if g_losses_epoch:
            self.history['g_loss'].append(np.mean(g_losses_epoch))
        
        print("Training completed!")
        return np.mean(d_losses), np.mean(g_losses) if g_losses else 0.0
    
    def train(self, dataloader, epochs=None, total_iterations=None, log_interval=None):
        """
        Train the GAN.
        
        Parameters
        ----------
        dataloader : DataLoader
            Training data loader.
        epochs : int, optional
            Number of epochs to train. Either epochs or total_iterations must be specified.
        total_iterations : int, optional
            Total number of iterations to train. Either epochs or total_iterations must be specified.
        log_interval : int, optional
            Interval to log losses when using total_iterations. If None and using epochs,
            logs every epoch. If None and using total_iterations, logs every len(dataloader) iterations.
        """
        if epochs is not None and total_iterations is not None:
            raise ValueError("Specify either epochs or total_iterations, not both")
        
        if epochs is None and total_iterations is None:
            raise ValueError("Must specify either epochs or total_iterations")
        
        if epochs is not None:
            # Convert epochs to iterations
            total_iterations = epochs * len(dataloader)
            if log_interval is None:
                log_interval = len(dataloader)  # Log every epoch
            
            print(f"Training for {epochs} epochs ({total_iterations} iterations)...")
            return self.train_iterations(dataloader, total_iterations, log_interval)
        else:
            # Train directly by iterations
            return self.train_iterations(dataloader, total_iterations, log_interval)
    
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
            img = samples[i]
            # Check if grayscale (single channel)
            if img.shape[0] == 1:
                img = img.squeeze(0).numpy()
                axes[i].imshow(img, cmap='gray')
            else:
                img = img.permute(1, 2, 0).numpy()
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

def create_trainer(generator_config=None, discriminator_config=None, trainer_config=None, g_optimizer=None, d_optimizer=None):
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
    g_optimizer : torch.optim.Optimizer
        Generator optimizer (must be provided).
    d_optimizer : torch.optim.Optimizer
        Discriminator optimizer (must be provided).
        
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
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        **trainer_config
    )
    
    return trainer