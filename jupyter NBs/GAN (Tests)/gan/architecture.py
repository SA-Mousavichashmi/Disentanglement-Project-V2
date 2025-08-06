# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    """Generator network based on Locatello decoder architecture with LeakyReLU activations."""
    
    def __init__(self, latent_dim=10, img_size=(3, 64, 64), use_batchnorm=True, negative_slope=0):
        """
        Initialize the Generator.
        
        Parameters
        ----------
        latent_dim : int
            Dimensionality of input noise vector.
        img_size : tuple of ints
            Size of output images. E.g. (3, 64, 64).
        use_batchnorm : bool
            Whether to use batch normalization layers.
        negative_slope : float
            Negative slope for LeakyReLU activation.
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope
        
        # Layer parameters
        kernel_size = 4
        n_chan = self.img_size[0]
        
        # Shape required to start transpose convs
        self.reshape = (64, kernel_size, kernel_size)
        
        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, 256, bias=not self.use_batchnorm)
        self.bn_lin1 = nn.BatchNorm1d(256) if self.use_batchnorm else nn.Identity()
        
        self.lin2 = nn.Linear(256, np.prod(self.reshape))
        
        # Transposed convolutional layers
        cnn_kwargs = dict(stride=2, padding=1, bias=not self.use_batchnorm)
        self.convT1 = nn.ConvTranspose2d(64, 64, kernel_size, **cnn_kwargs)
        self.bn1 = nn.BatchNorm2d(64) if self.use_batchnorm else nn.Identity()
        
        self.convT2 = nn.ConvTranspose2d(64, 32, kernel_size, **cnn_kwargs)
        self.bn2 = nn.BatchNorm2d(32) if self.use_batchnorm else nn.Identity()
        
        self.convT3 = nn.ConvTranspose2d(32, 32, kernel_size, **cnn_kwargs)
        self.bn3 = nn.BatchNorm2d(32) if self.use_batchnorm else nn.Identity()
        
        # Last layer keeps bias so pixels can shift freely; no BN here
        self.convT4 = nn.ConvTranspose2d(32, n_chan, kernel_size, stride=2, padding=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        """
        Forward pass through generator.
        
        Parameters
        ----------
        z : torch.Tensor
            Input noise vector of shape (batch_size, latent_dim).
            
        Returns
        -------
        torch.Tensor
            Generated images of shape (batch_size, *img_size).
        """
        batch_size = z.size(0)
        
        # Fully connected layers with LeakyReLU activations
        x = F.leaky_relu(self.bn_lin1(self.lin1(z)), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.lin2(x), negative_slope=self.negative_slope)
        x = x.view(batch_size, *self.reshape)
        
        # Transposed convolutional layers with LeakyReLU activations
        x = F.leaky_relu(self.bn1(self.convT1(x)), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.bn2(self.convT2(x)), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.bn3(self.convT3(x)), negative_slope=self.negative_slope)
        
        # Final layer with Tanh activation to output images in [-1, 1]
        # x = torch.tanh(self.convT4(x))

        # Final layer using sigmoid activation to output images in [0, 1]
        x = torch.sigmoid(self.convT4(x))

        return x


class Discriminator(nn.Module):
    """Discriminator network based on Locatello encoder architecture with optional spectral normalization."""
    
    def __init__(self, img_size=(3, 64, 64), use_batchnorm=False, use_spectral_norm=False, negative_slope=0.2):
        """
        Initialize the Discriminator.
        
        Parameters
        ----------
        img_size : tuple of ints
            Size of input images. E.g. (3, 64, 64).
        use_batchnorm : bool
            Whether to use batch normalization layers.
        use_spectral_norm : bool
            Whether to use spectral normalization on convolutional layers.
        negative_slope : float
            Negative slope for LeakyReLU activation.
        """
        super(Discriminator, self).__init__()
        
        self.img_size = img_size
        self.use_batchnorm = use_batchnorm
        self.use_spectral_norm = use_spectral_norm
        self.negative_slope = negative_slope
        
        # Layer parameters
        kernel_size = 4
        n_chan = self.img_size[0]
        
        assert_str = "This architecture requires 64x64 inputs."
        assert self.img_size[-2] == self.img_size[-1] == 64, assert_str
        
        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1, bias=not self.use_batchnorm)
        
        # Build convolutional layers with optional spectral normalization
        self.conv1 = nn.Conv2d(n_chan, 32, kernel_size, **cnn_kwargs)
        if self.use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
        self.bn1 = nn.BatchNorm2d(32) if self.use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size, **cnn_kwargs)
        if self.use_spectral_norm:
            self.conv2 = spectral_norm(self.conv2)
        self.bn2 = nn.BatchNorm2d(32) if self.use_batchnorm else nn.Identity()
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size, **cnn_kwargs)
        if self.use_spectral_norm:
            self.conv3 = spectral_norm(self.conv3)
        self.bn3 = nn.BatchNorm2d(64) if self.use_batchnorm else nn.Identity()
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size, **cnn_kwargs)
        if self.use_spectral_norm:
            self.conv4 = spectral_norm(self.conv4)
        self.bn4 = nn.BatchNorm2d(64) if self.use_batchnorm else nn.Identity()
        
        # Fully connected layers
        self.lin = nn.Linear(int((64/(2**4))**2 * 64), 256)
        if self.use_spectral_norm:
            self.lin = spectral_norm(self.lin)
            
        # Final output layer (single value for real/fake classification)
        self.output = nn.Linear(256, 1)
        if self.use_spectral_norm:
            self.output = spectral_norm(self.output)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through discriminator.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch_size, *img_size).
            
        Returns
        -------
        torch.Tensor
            Discriminator output of shape (batch_size, 1).
        """
        batch_size = x.size(0)
        
        # Convolutional layers with LeakyReLU activations
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=self.negative_slope)
        
        # Fully connected layers with LeakyReLU activation
        x = x.view((batch_size, -1))
        x = F.leaky_relu(self.lin(x), negative_slope=self.negative_slope)
        
        # Final output (no activation for flexibility with different losses)
        x = self.output(x)
        
        return x


def create_gan(generator_config=None, discriminator_config=None):
    """
    Create a GAN with specified generator and discriminator configurations.
    
    Parameters
    ----------
    generator_config : dict, optional
        Configuration dictionary for the generator.
    discriminator_config : dict, optional
        Configuration dictionary for the discriminator.
        
    Returns
    -------
    tuple
        (generator, discriminator) models.
    """
    if generator_config is None:
        generator_config = {}
    if discriminator_config is None:
        discriminator_config = {}
    
    generator = Generator(**generator_config)
    discriminator = Discriminator(**discriminator_config)
    
    return generator, discriminator
