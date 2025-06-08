import torch
import numpy as np
from vae_models.utils import get_device, decode_latents
from vae_models.n_vae.utils import get_traversal_range
from vae_models.s_vae.toroidal_vae.utils import get_toroidal_traversal_range

####################### R1/S1 Topology Latent Traversal Functions #########################

def traverse_single_r1_s1_latent(vae_model,
                                latent_factor_idx,
                                max_traversal_type=None,
                                max_traversal=None,
                                num_samples=10,
                                ref_img=None
                                ):
    """
    Latent traversal for single R1/S1 topology latent dimension.
    Automatically determines whether to use R1 or S1 traversal based on the factor topology.

    Parameters
    ----------
    vae_model : torch.nn.Module
        The S-N VAE model with R1/S1 topology latent factors.
    latent_factor_idx : int
        The index of the latent factor to traverse.    max_traversal_type : str, optional
        Specifies how the traversal range is determined.
        - 'probability' or 'absolute': Used for R1 factors
        - 'fraction': Used for S1 factors
        If None, defaults are used (both parameters must be None together).
        Defaults: 'probability' for R1, 'fraction' for S1.
    max_traversal : float, optional
        The maximum traversal value, interpreted based on `max_traversal_type`.
        If None, defaults are used (both parameters must be None together).
        Defaults: 0.99 for R1, 1.0 for S1.
    num_samples : int, optional
        The number of steps or images to generate along the traversal. Defaults to 10.
    ref_img : torch.Tensor, optional
        A reference image tensor (C, H, W) to base the traversal on. If None,
        traversal is centered around the prior. Defaults to None.

    Returns
    -------
    torch.Tensor
        A tensor containing the generated images corresponding to the traversal.
        Shape (num_samples, C, H, W).    """
    assert latent_factor_idx in range(vae_model.latent_factor_num), f"latent_factor_idx must be in range [0, {vae_model.latent_factor_num-1}]"
    
    if not hasattr(vae_model, 'latent_factor_topologies'):
        raise ValueError("VAE model must have 'latent_factor_topologies' attribute for R1/S1 topology traversal")
    
    # Validate that both parameters are either None or both are provided
    if (max_traversal_type is None) != (max_traversal is None):
        raise ValueError("Both max_traversal_type and max_traversal must be either both None (use defaults) or both provided.")
    
    # Determine the topology of the requested latent factor
    factor_topology = vae_model.latent_factor_topologies[latent_factor_idx]
    
    # Set default values based on topology if both parameters are None
    if max_traversal_type is None and max_traversal is None:
        if factor_topology == 'R1':
            max_traversal_type = 'probability'
            max_traversal = 0.99
        elif factor_topology == 'S1':
            max_traversal_type = 'fraction'
            max_traversal = 1.0
        else:
            raise ValueError(f"Unknown latent factor topology: {factor_topology}")
        
    if factor_topology == 'R1':
        return _traverse_single_r1_factor(
            vae_model=vae_model,
            latent_factor_idx=latent_factor_idx,
            num_samples=num_samples,
            max_traversal_type=max_traversal_type,
            max_traversal=max_traversal,
            ref_img=ref_img
        )
    elif factor_topology == 'S1':
        return _traverse_single_s1_factor(
            vae_model=vae_model,
            latent_factor_idx=latent_factor_idx,
            num_samples=num_samples,
            max_traversal_type=max_traversal_type,
            max_traversal=max_traversal,
            ref_img=ref_img
        )
    else:
        raise ValueError(f"Unknown latent factor topology: {factor_topology}")


def traverse_all_r1_s1_latents(vae_model,
                              r1_max_traversal_type='probability',
                              r1_max_traversal=0.99,
                              s1_max_traversal_type='fraction',
                              s1_max_traversal=1.0,
                              num_samples=10,
                              ref_img=None
                              ):
    """
    Latent traversal for all R1/S1 topology latent dimensions.

    Parameters
    ----------
    vae_model : torch.nn.Module
        The S-N VAE model with R1/S1 topology latent factors.    r1_max_traversal_type : str, optional
        Specifies how the traversal range is determined for R1 factors.
        - 'probability' or 'absolute': Used for R1 factors. Defaults to 'probability'.
    r1_max_traversal : float, optional
        The maximum traversal value for R1 factors, interpreted based on `r1_max_traversal_type`.
        Defaults to 0.99.
    s1_max_traversal_type : str, optional
        Specifies how the traversal range is determined for S1 factors.
        - 'fraction': Used for S1 factors. Defaults to 'fraction'.
    s1_max_traversal : float, optional
        The maximum traversal value for S1 factors, interpreted based on `s1_max_traversal_type`.
        Defaults to 1.0.
    num_samples : int, optional
        The number of steps or images to generate along each traversal. Defaults to 10.
    ref_img : torch.Tensor, optional
        A reference image tensor (C, H, W) to base the traversal on. If None,
        traversal is centered around the prior for all dimensions. Defaults to None.

    Returns
    -------
    list of torch.Tensor
        A list containing tensors of generated images for each latent factor.
        Each tensor has shape (num_samples, C, H, W).
    """
    # Validate parameter compatibility with topologies
    if r1_max_traversal_type not in ['probability', 'absolute']:
        raise ValueError(f"r1_max_traversal_type '{r1_max_traversal_type}' is not valid for R1 topology. Use 'probability' or 'absolute'.")
    
    if s1_max_traversal_type != 'fraction':
        raise ValueError(f"s1_max_traversal_type '{s1_max_traversal_type}' is not valid for S1 topology. Use 'fraction'.")
    
    all_traversals = []
    for latent_factor_idx in range(vae_model.latent_factor_num):
        # Determine the topology of the current latent factor
        factor_topology = vae_model.latent_factor_topologies[latent_factor_idx]
        
        # Select appropriate parameters based on topology
        if factor_topology == 'R1':
            max_traversal_type = r1_max_traversal_type
            max_traversal = r1_max_traversal
        elif factor_topology == 'S1':
            max_traversal_type = s1_max_traversal_type
            max_traversal = s1_max_traversal
        else:
            raise ValueError(f"Unknown latent factor topology: {factor_topology}")
        
        traversal_images = traverse_single_r1_s1_latent(
            vae_model=vae_model,
            latent_factor_idx=latent_factor_idx,
            max_traversal_type=max_traversal_type,
            max_traversal=max_traversal,
            num_samples=num_samples,
            ref_img=ref_img
        )
        all_traversals.append(traversal_images)
    return all_traversals


def _traverse_single_r1_factor(vae_model, latent_factor_idx, num_samples, max_traversal_type, max_traversal, ref_img):
    """Traverse a single R1 (normal) latent factor using R1/S1 topology approach."""
    device = get_device(vae_model)
    
    if ref_img is not None:
        # Ensure ref_img is on the correct device and has batch dimension
        if ref_img.dim() == 3:  # Assuming (C, H, W)
            ref_img = ref_img.unsqueeze(0)  # Add batch dim -> (1, C, H, W)
        ref_img = ref_img.to(device)

        with torch.no_grad():
            # Get latent representation for the reference image
            latent_repr = vae_model.get_representations(ref_img, is_deterministic=True)
            
            # Parse the latent representation to extract the R1 component for this factor
            r1_component = _extract_r1_component(vae_model, latent_repr, latent_factor_idx)
            base_latent = latent_repr
            mean_val = r1_component.item()
            
            # For R1 factors, use standard deviation of 1.0 (could be made configurable)
            std_dev = 1.0
            
    else:
        # Use zeros as base representation
        total_latent_dim = vae_model.total_latent_dim
        base_latent = torch.zeros(1, total_latent_dim, device=device)
        mean_val = 0.0
        std_dev = 1.0
        
    # Get the traversal range for the R1 factor
    min_val, max_val = get_traversal_range(
        max_traversal_type=max_traversal_type,
        max_traversal=max_traversal,
        mean=mean_val,
        std=std_dev
    )
    
    # Create traversal values
    traversal_values = torch.linspace(min_val, max_val, num_samples, device=device)
    
    # Create latent vectors for traversal
    latent_vectors = base_latent.repeat(num_samples, 1)
    
    # Modify only the R1 component for the specified factor
    r1_start_idx = _get_r1_factor_start_index(vae_model, latent_factor_idx)
    latent_vectors[:, r1_start_idx] = traversal_values
    
    # Decode the latent vectors into images
    generated_images = decode_latents(vae_model, latent_vectors)
    return generated_images


def _traverse_single_s1_factor(vae_model, latent_factor_idx, num_samples, max_traversal_type, max_traversal, ref_img):
    """Traverse a single S1 (spherical) latent factor using R1/S1 topology approach."""
    device = get_device(vae_model)
    
    if ref_img is not None:
        # Ensure ref_img is on the correct device and has batch dimension
        if ref_img.dim() == 3:  # Assuming (C, H, W)
            ref_img = ref_img.unsqueeze(0)  # Add batch dim -> (1, C, H, W)
        ref_img = ref_img.to(device)

        with torch.no_grad():
            # Get latent representation for the reference image
            latent_repr = vae_model.get_representations(ref_img, is_deterministic=True)
            
            # Extract the S1 component for this factor (2D coordinates on unit circle)
            s1_component = _extract_s1_component(vae_model, latent_repr, latent_factor_idx)
            
            # Convert S1 coordinates (cos(θ), sin(θ)) to angle θ
            cos_val = s1_component[0].item()
            sin_val = s1_component[1].item()
            center_angle = torch.atan2(torch.tensor(sin_val), torch.tensor(cos_val)).item()
            
            # Normalize angle to [0, 2π) for consistency
            center_angle = (center_angle + 2 * np.pi) % (2 * np.pi)
            
            base_latent = latent_repr
            
    else:
        # Use zeros as base representation
        total_latent_dim = vae_model.total_latent_dim
        base_latent = torch.zeros(1, total_latent_dim, device=device)
        center_angle = 0.0
        
    # Get the traversal range for the S1 factor
    min_angle, max_angle = get_toroidal_traversal_range(
        max_traversal_type=max_traversal_type,
        max_traversal=max_traversal,
        center_angle=center_angle
    )
    
    # Create traversal angles
    traversal_angles = torch.linspace(min_angle, max_angle, num_samples, device=device)
    
    # Create latent vectors for traversal
    latent_vectors = base_latent.repeat(num_samples, 1)
    
    # Modify only the S1 component for the specified factor
    s1_start_idx = _get_s1_factor_start_index(vae_model, latent_factor_idx)
    cos_vals = torch.cos(traversal_angles)
    sin_vals = torch.sin(traversal_angles)
    
    latent_vectors[:, s1_start_idx] = cos_vals
    latent_vectors[:, s1_start_idx + 1] = sin_vals
    
    # Decode the latent vectors into images
    generated_images = decode_latents(vae_model, latent_vectors)
    return generated_images


def _extract_r1_component(vae_model, latent_repr, factor_idx):
    """Extract the R1 component for a specific factor from the latent representation."""
    latent_start_idx = 0
    
    for i, topology in enumerate(vae_model.latent_factor_topologies):
        if i == factor_idx:
            if topology != 'R1':
                raise ValueError(f"Factor {factor_idx} is not R1 topology")
            return latent_repr[:, latent_start_idx]
        
        # Move to next factor's start position
        if topology == 'R1':
            latent_start_idx += 1
        elif topology == 'S1':
            latent_start_idx += 2
    
    raise ValueError(f"Factor index {factor_idx} out of range")


def _extract_s1_component(vae_model, latent_repr, factor_idx):
    """Extract the S1 component (2D coordinates) for a specific factor from the latent representation."""
    latent_start_idx = 0
    
    for i, topology in enumerate(vae_model.latent_factor_topologies):
        if i == factor_idx:
            if topology != 'S1':
                raise ValueError(f"Factor {factor_idx} is not S1 topology")
            return latent_repr[:, latent_start_idx:latent_start_idx+2].squeeze(0)
        
        # Move to next factor's start position
        if topology == 'R1':
            latent_start_idx += 1
        elif topology == 'S1':
            latent_start_idx += 2
    
    raise ValueError(f"Factor index {factor_idx} out of range")


def _get_r1_factor_start_index(vae_model, factor_idx):
    """Get the starting index in the latent vector for an R1 factor."""
    latent_start_idx = 0
    
    for i, topology in enumerate(vae_model.latent_factor_topologies):
        if i == factor_idx:
            if topology != 'R1':
                raise ValueError(f"Factor {factor_idx} is not R1 topology")
            return latent_start_idx
        
        # Move to next factor's start position
        if topology == 'R1':
            latent_start_idx += 1
        elif topology == 'S1':
            latent_start_idx += 2
    
    raise ValueError(f"Factor index {factor_idx} out of range")


def _get_s1_factor_start_index(vae_model, factor_idx):
    """Get the starting index in the latent vector for an S1 factor."""
    latent_start_idx = 0
    
    for i, topology in enumerate(vae_model.latent_factor_topologies):
        if i == factor_idx:
            if topology != 'S1':
                raise ValueError(f"Factor {factor_idx} is not S1 topology")
            return latent_start_idx
        
        # Move to next factor's start position
        if topology == 'R1':
            latent_start_idx += 1
        elif topology == 'S1':
            latent_start_idx += 2
    
    raise ValueError(f"Factor index {factor_idx} out of range")