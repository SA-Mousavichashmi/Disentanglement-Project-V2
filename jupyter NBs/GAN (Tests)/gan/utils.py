"""
Utility functions for GAN training and visualization.

This module contains helper functions for data visualization, memory management,
and other utilities used across GAN training notebooks and scripts.
"""

import matplotlib.pyplot as plt
import random
import torch
import gc
import numpy as np


def show_dataset_samples(dataset, n_samples=16, figsize=(6, 6), title=None):
    """
    Display a grid of sample images from any dataset.
    
    Automatically detects whether images are RGB (3-channel) or grayscale (1-channel)
    and displays them appropriately.
    
    Args:
        dataset: Dataset object with __getitem__ method returning (image, factors)
        n_samples (int): Number of samples to display (default: 16)
        figsize (tuple): Figure size as (width, height) (default: (6, 6))
        title (str): Title for the plot (default: auto-generated)
    """
    # Calculate grid dimensions
    n_cols = int(np.sqrt(n_samples))
    n_rows = int(np.ceil(n_samples / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Generate n_samples random indices
    indices = random.sample(range(len(dataset)), n_samples)
    
    for i, idx in enumerate(indices):
        img, factors = dataset[idx]
        
        # Detect image format and convert appropriately
        if img.dim() == 3:  # (C, H, W)
            if img.shape[0] == 1:  # Grayscale
                img_np = img.squeeze().numpy()
                axes[i].imshow(img_np, cmap='gray')
            elif img.shape[0] == 3:  # RGB
                img_np = img.permute(1, 2, 0).numpy()
                axes[i].imshow(img_np)
            else:
                raise ValueError(f"Unsupported number of channels: {img.shape[0]}")
        else:
            raise ValueError(f"Expected 3D tensor (C, H, W), got {img.dim()}D")
        
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}', fontsize=8)
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Set title
    if title is None:
        title = f'Dataset Samples (n={n_samples})'
    fig.suptitle(title, fontsize=16, y=1.02, ha='center')
    plt.show()


def free_memory():
    """
    Release RAM and CUDA memory occupied by tensors, models, and data loaders.
    
    This function runs the Python garbage collector and empties the CUDA cache
    to free up memory resources.
    """
    # Run Python garbage collector
    gc.collect()

    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("RAM and CUDA memory have been freed.")


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducible results.
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For consistent results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seeds set to {seed} for reproducibility.")
