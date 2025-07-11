"""
GAN Discriminator Architectures for Group Theory Implementation.
This module provides direct discriminator implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any


class LocatelloDiscriminator(nn.Module):
    """Standard discriminator based on Locatello encoder architecture."""
    
    def __init__(self, input_channels: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        
        kernel_size = 4
        cnn_kwargs = dict(stride=2, padding=1)
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(32, 32, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(32, 64, kernel_size, **cnn_kwargs)
        self.conv4 = nn.Conv2d(64, 64, kernel_size, **cnn_kwargs)
        
        self.lin = nn.Linear(int((64/(2**4))**2 * 64), hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.view(batch_size, -1)
        x = F.leaky_relu(self.lin(x), 0.2)
        x = self.dropout(x)
        return self.output_layer(x)


class SpectralNormDiscriminator(nn.Module):
    """Spectral normalized discriminator for SN-GAN."""
    
    def __init__(self, input_channels: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        
        kernel_size = 4
        cnn_kwargs = dict(stride=2, padding=1)
        
        # Apply spectral normalization to all layers
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(input_channels, 32, kernel_size, **cnn_kwargs))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(32, 32, kernel_size, **cnn_kwargs))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size, **cnn_kwargs))
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(64, 64, kernel_size, **cnn_kwargs))
        
        self.lin = nn.utils.spectral_norm(nn.Linear(int((64/(2**4))**2 * 64), hidden_dim))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_layer = nn.utils.spectral_norm(nn.Linear(hidden_dim, 1))
    
    def forward(self, x):
        batch_size = x.size(0)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.view(batch_size, -1)
        x = F.leaky_relu(self.lin(x), 0.2)
        x = self.dropout(x)
        return self.output_layer(x)


_discriminators = {
    "locatello": LocatelloDiscriminator,
    "spectral_norm": SpectralNormDiscriminator,
}


def select_discriminator(architecture_type: str, input_channels: int, **kwargs) -> nn.Module:
    """
    Select and create a discriminator network based on architecture type.
    
    Args:
        architecture_type (str): The type of discriminator architecture ("locatello", "spectral_norm").
        input_channels (int): Number of input channels.
        **kwargs: Additional arguments for discriminator constructor.
        
    Returns:
        nn.Module: The discriminator network.
        
    Raises:
        ValueError: If an unknown architecture type is provided.
    """
    if architecture_type not in _discriminators:
        available = list(_discriminators.keys())
        raise ValueError(f"Unknown architecture type: {architecture_type}. Available: {available}")
    
    return _discriminators[architecture_type](input_channels, **kwargs)
