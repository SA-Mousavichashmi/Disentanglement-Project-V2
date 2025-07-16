import torch
import numpy as np
from vae_models.utils import get_device, decode_latents

####################### Latent Traversal Functions #########################

def get_toroidal_traversal_range(max_traversal_type, max_traversal, center_angle=0):
    """Calculates the traversal range for toroidal latent space based on the specified type and value.
    
    For toroidal VAE, the latent space is on S1 circles, so angles are in [0, 2π).
    Traversal is centered around a given angle and wraps around the circle.

    Parameters
    ----------
    max_traversal_type : str
        Specifies how the traversal range is determined ('fraction' or 'absolute').
    max_traversal : float
        The maximum traversal value, interpreted based on `max_traversal_type`.
        If 'fraction', it's the fraction of the full circle to cover (e.g., 0.5 for half circle).
        If 'absolute', it's the absolute angular range in radians from the center.
    center_angle : float, optional
        The center angle for the traversal in radians. Defaults to 0.

    Returns
    -------
    tuple of (float, float)
        A tuple containing the minimum and maximum traversal angles in radians.
    """
    if max_traversal_type == 'fraction':
        if not (0 < max_traversal <= 1):
            raise ValueError("max_traversal must be in the range (0, 1] for 'fraction' type.")
        # Calculate the angular range based on fraction of full circle
        angular_range = max_traversal * 2 * np.pi
        half_range = angular_range / 2
    elif max_traversal_type == 'absolute':
        if max_traversal <= 0:
            raise ValueError("max_traversal must be positive for 'absolute' type.")
        half_range = max_traversal
    else:
        raise ValueError(f"Unknown max_traversal_type: {max_traversal_type}")

    # Calculate min and max angles
    min_angle = center_angle - half_range
    max_angle = center_angle + half_range
    
    return (min_angle, max_angle)

def traverse_single_toroidal_latent(vae_model,
                                   latent_factor_idx,
                                   num_samples=10,
                                   max_traversal_type='fraction',
                                   max_traversal=0.5,
                                   ref_img=None
                                   ):
    """
    Latent traversal for single toroidal latent dimension.
    If ref_img is provided, traversal is centered around its encoded angles.

    Parameters
    ----------
    vae_model : torch.nn.Module
        The toroidal VAE model.
    latent_factor_idx : int
        The index of the latent dimension to traverse.
    num_samples : int, optional
        The number of steps or images to generate along the traversal. Defaults to 10.    max_traversal_type : str
        Specifies how the traversal range is determined ('fraction' or 'absolute').
    max_traversal : float
        The maximum traversal value, interpreted based on `max_traversal_type`.
    ref_img : torch.Tensor, optional
        A reference image tensor (C, H, W) to base the traversal on. If None,
        traversal is centered around angle 0. Defaults to None.

    Returns
    -------
    torch.Tensor
        A tensor containing the generated images corresponding to the traversal.
        Shape (num_samples, C, H, W).
    """
    assert latent_factor_idx in range(vae_model.latent_factor_num), f"latent_factor_idx must be in range [0, {vae_model.latent_factor_num-1}]"
    device = get_device(vae_model)

    if ref_img is not None:
        # Ensure ref_img is on the correct device and has batch dimension
        if ref_img.dim() == 3:  # Assuming (C, H, W)
            ref_img = ref_img.unsqueeze(0)  # Add batch dim -> (1, C, H, W)
        ref_img = ref_img.to(device)

        with torch.no_grad():
            # Use the model's method to get deterministic latent representations
            # This returns the mean (mu) values which are points on S1 for each latent factor
            latent_repr = vae_model.get_representations(ref_img, is_deterministic=True)
            # latent_repr shape: (batch_size, latent_factor_num * 2)
            
            # Reshape to (batch_size, latent_factor_num, 2) to separate each S1 coordinate
            latent_repr_reshaped = latent_repr.view(-1, vae_model.latent_factor_num, 2)
            
            # Convert S1 coordinates (cos(θ), sin(θ)) to angles θ using atan2
            cos_vals = latent_repr_reshaped[:, :, 0]  # cos(θ) values
            sin_vals = latent_repr_reshaped[:, :, 1]  # sin(θ) values
            angles = torch.atan2(sin_vals, cos_vals)  # θ values in [-π, π]
            
            # Normalize angles to [0, 2π) for consistency
            angles = (angles + 2 * np.pi) % (2 * np.pi)

        # Use the encoded angles as the base
        base_angles = angles
        center_angle = angles[:, latent_factor_idx].item()
          # Get the traversal range
        min_angle, max_angle = get_toroidal_traversal_range(
            max_traversal_type=max_traversal_type,
            max_traversal=max_traversal,
            center_angle=center_angle
        )
    else:
        # Use default center (angle 0) for all dimensions
        base_angles = torch.zeros(1, vae_model.latent_factor_num, device=device)
        center_angle = 0.0
        
        # Get the traversal range
        min_angle, max_angle = get_toroidal_traversal_range(
            max_traversal_type=max_traversal_type,
            max_traversal=max_traversal,
            center_angle=center_angle
        )

    # Create traversal angles
    traversal_angles = torch.linspace(min_angle, max_angle, num_samples, device=device)

    # Create latent vectors for traversal
    latent_angles = base_angles.repeat(num_samples, 1)
    latent_angles[:, latent_factor_idx] = traversal_angles

    # Convert angles to the format expected by the decoder
    # The toroidal VAE decoder expects latent vectors in (cos, sin) format
    # Shape: (num_samples, latent_factor_num * 2)
    cos_vals = torch.cos(latent_angles)
    sin_vals = torch.sin(latent_angles)
    
    # Interleave cos and sin values to match the expected format
    # For each latent factor i: [cos(θ_i), sin(θ_i)]
    latent_vectors = torch.stack([cos_vals, sin_vals], dim=-1)  # Shape: (num_samples, latent_factor_num, 2)
    latent_vectors = latent_vectors.flatten(start_dim=1)  # Shape: (num_samples, latent_factor_num * 2)

    # Decode the latent vectors into images
    generated_images = decode_latents(vae_model, latent_vectors)
    return generated_images

def traverse_all_toroidal_latents(vae_model,
                                 num_samples=10,
                                 max_traversal_type='fraction',
                                 max_traversal=0.5,
                                 ref_img=None
                                 ):
    """
    Latent traversal for all toroidal latent dimensions.

    Parameters
    ----------
    vae_model : torch.nn.Module
        The toroidal VAE model.
    num_samples : int, optional
        The number of steps or images to generate along each traversal. Defaults to 10.    max_traversal_type : str
        Specifies how the traversal range is determined ('fraction' or 'absolute').
    max_traversal : float
        The maximum traversal value, interpreted based on `max_traversal_type`.
    ref_img : torch.Tensor, optional
        A reference image tensor (C, H, W) to base the traversal on. If None,
        traversal is centered around angle 0 for all dimensions. Defaults to None.

    Returns
    -------    list of torch.Tensor
        A list containing tensors of generated images for each latent dimension.
        Each tensor has shape (num_samples, C, H, W).
    """
    all_traversals = []
    for latent_idx in range(vae_model.latent_factor_num):
        traversal_images = traverse_single_toroidal_latent(
            vae_model=vae_model,
            latent_factor_idx=latent_idx,
            num_samples=num_samples,
            max_traversal_type=max_traversal_type,
            max_traversal=max_traversal,
            ref_img=ref_img
        )
        all_traversals.append(traversal_images)
    return all_traversals
