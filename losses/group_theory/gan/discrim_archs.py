"""
GAN Discriminator Architectures for Group Theory Implementation.
This module provides discriminator implementations with a base class approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Union


class BaseDiscriminator(nn.Module):
    """Base discriminator class with configurable normalization and other parameters."""
    
    def __init__(
        self, 
        input_channels: int, 
        hidden_dim: int = 256, 
        dropout: float = 0.0,
        spectral_norm: bool = False,
        activation_slope: float = 0.2,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1
    ):
        """
        Initialize base discriminator.
        
        Args:
            input_channels (int): Number of input channels
            hidden_dim (int): Hidden dimension size
            dropout (float): Dropout rate
            spectral_norm (bool): Whether to apply spectral normalization
            activation_slope (float): Negative slope for LeakyReLU
            kernel_size (int): Convolution kernel size
            stride (int): Convolution stride
            padding (int): Convolution padding
        """
        super().__init__()
        
        self.spectral_norm = spectral_norm
        self.activation_slope = activation_slope
        self.dropout_rate = dropout
        
        # Store conv parameters
        self.kernel_size = kernel_size
        self.cnn_kwargs = dict(stride=stride, padding=padding)
        
        # Initialize layers (to be implemented by subclasses)
        self._build_layers(input_channels, hidden_dim)
        
    def _apply_normalization(self, layer: nn.Module) -> nn.Module:
        """Apply normalization to a layer if specified."""
        if self.spectral_norm:
            return nn.utils.spectral_norm(layer)
        return layer
    
    def _build_layers(self, input_channels: int, hidden_dim: int):
        """Build discriminator layers. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _build_layers method")
    
    def forward(self, x):
        """Forward pass. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")


class LocatelloDiscriminator(BaseDiscriminator):
    """Standard discriminator based on Locatello encoder architecture."""
    
    def __init__(
        self, 
        input_channels: int, 
        hidden_dim: int = 256, 
        dropout: float = 0.0,
        spectral_norm: bool = False,
        **kwargs
    ):
        """
        Initialize Locatello discriminator.
        
        Args:
            input_channels (int): Number of input channels
            hidden_dim (int): Hidden dimension size  
            dropout (float): Dropout rate
            spectral_norm (bool): Whether to apply spectral normalization
            **kwargs: Additional arguments for base class
        """
        super().__init__(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            spectral_norm=spectral_norm,
            **kwargs
        )
    
    def _build_layers(self, input_channels: int, hidden_dim: int):
        """Build Locatello discriminator layers."""
        # Convolutional layers
        self.conv1 = self._apply_normalization(
            nn.Conv2d(input_channels, 32, self.kernel_size, **self.cnn_kwargs)
        )
        self.conv2 = self._apply_normalization(
            nn.Conv2d(32, 32, self.kernel_size, **self.cnn_kwargs)
        )
        self.conv3 = self._apply_normalization(
            nn.Conv2d(32, 64, self.kernel_size, **self.cnn_kwargs)
        )
        self.conv4 = self._apply_normalization(
            nn.Conv2d(64, 64, self.kernel_size, **self.cnn_kwargs)
        )
        
        # Linear layers
        linear_input_size = int((64/(2**4))**2 * 64)
        self.lin = self._apply_normalization(
            nn.Linear(linear_input_size, hidden_dim)
        )
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        self.output_layer = self._apply_normalization(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """Forward pass through Locatello discriminator."""
        batch_size = x.size(0)
        x = F.leaky_relu(self.conv1(x), self.activation_slope)
        x = F.leaky_relu(self.conv2(x), self.activation_slope)
        x = F.leaky_relu(self.conv3(x), self.activation_slope)
        x = F.leaky_relu(self.conv4(x), self.activation_slope)
        x = x.view(batch_size, -1)
        x = F.leaky_relu(self.lin(x), self.activation_slope)
        x = self.dropout(x)
        return self.output_layer(x)


_discriminators = {
    "locatello": LocatelloDiscriminator,
}


def select_discriminator(
    architecture_type: str, 
    input_channels: int, 
    spectral_norm: bool = False,
    **kwargs
) -> nn.Module:
    """
    Select and create a discriminator network based on architecture type.
    
    Args:
        architecture_type (str): The type of discriminator architecture ("locatello").
        input_channels (int): Number of input channels.
        spectral_norm (bool): Whether to apply spectral normalization.
        **kwargs: Additional arguments for discriminator constructor.
        
    Returns:
        nn.Module: The discriminator network.
        
    Raises:
        ValueError: If an unknown architecture type is provided.
        
    Examples:
        # Standard Locatello discriminator
        disc = select_discriminator("locatello", input_channels=3)
        
        # Locatello discriminator with spectral normalization (equivalent to old SN-GAN)
        disc = select_discriminator("locatello", input_channels=3, spectral_norm=True)
        
        # Locatello discriminator with custom parameters
        disc = select_discriminator("locatello", input_channels=3, hidden_dim=512, dropout=0.1)
    """
    if architecture_type not in _discriminators:
        available = list(_discriminators.keys())
        raise ValueError(f"Unknown architecture type: {architecture_type}. Available: {available}")
    
    return _discriminators[architecture_type](
        input_channels=input_channels, 
        spectral_norm=spectral_norm,
        **kwargs
    )


# Export all important classes and functions
__all__ = [
    'select_discriminator',
]
