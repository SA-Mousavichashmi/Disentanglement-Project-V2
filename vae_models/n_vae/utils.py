import torch
from scipy import stats  # Add scipy import
from vae_models.utils import get_device, decode_latents

####################### Latent Traversal Functions #########################

def get_traversal_range(max_traversal_type, max_traversal, mean=0, std=1):
    """Calculates the traversal range based on the specified type and value, centered around the mean.

    Parameters
    ----------
    max_traversal_type : str
        Specifies how the traversal range is determined ('probability' or 'absolute').
    max_traversal : float
        The maximum traversal value, interpreted based on `max_traversal_type`.
        If 'probability', it's the probability mass to cover symmetrically around the mean (e.g., 0.95 for 95%).
        If 'absolute', it's the absolute deviation from the mean.
    mean : float, optional
        The mean of the latent variable distribution. Defaults to 0.
    std : float, optional
        The standard deviation of the latent variable distribution. Defaults to 1.

    Returns
    -------
    tuple of (float, float)
        A tuple containing the minimum and maximum traversal values, centered around the mean.
    """
    if max_traversal < 0 and max_traversal > 1 and max_traversal_type == 'probability':
        raise ValueError("max_traversal must be in the range [0, 1] for 'probability' type.")

    if max_traversal_type == 'probability':
        # Calculate the z-score corresponding to the upper limit of the central probability mass
        # e.g., for 95% (max_traversal=0.95), we need the ppf of 0.5 + 0.95/2 = 0.975
        z_score = stats.norm.ppf(0.5 + max_traversal / 2)
        # Calculate the deviation from the mean based on the z-score and std
        deviation = z_score * std
    elif max_traversal_type == 'absolute':
        # max_traversal directly specifies the absolute deviation from the mean
        deviation = max_traversal
    else:
        raise ValueError(f"Unknown max_traversal_type: {max_traversal_type}")

    # Return the range symmetrically around the mean
    return (mean - deviation, mean + deviation)

def traverse_single_latent(vae_model,
                           latent_factor_idx,
                           num_samples=10,
                           max_traversal_type='probability',
                           max_traversal=0.95,
                           ref_img=None,
                           use_ref_img_lat_std=False # Add parameter
                           ):
    """
    Latent traversal for single latent dim based on the traversal range.
    If ref_img is provided, traversal is based on its encoded mean and variance.
    If use_ref_img_lat_std is True, the std dev from the encoder is used, otherwise std=1.

    Parameters
    ----------
    vae_model : torch.nn.Module
        The VAE model.
    latent_factor_idx : int
        The index of the latent dimension to traverse.
    max_traversal_type : str
        Specifies how the traversal range is determined ('probability' or 'absolute').
    max_traversal : float
        The maximum traversal value, interpreted based on `max_traversal_type`.
    num_samples : int, optional
        The number of steps or images to generate along the traversal. Defaults to 10.
    ref_img : torch.Tensor, optional
        A reference image tensor (C, H, W) to base the traversal on. If None,
        traversal is based on the prior (mean=0, std=1). Defaults to None.
    use_ref_img_lat_std : bool, optional
        If True and ref_img is provided, use the standard deviation from the
        encoder's variance output for the specified latent dimension.
        Otherwise, use std=1. Defaults to False.

    Returns
    -------
    torch.Tensor
        A tensor containing the generated images corresponding to the traversal.
        Shape (num_samples, C, H, W).
    """
    assert latent_factor_idx in range(vae_model.latent_dim), f"latent_factor_idx must be in range [0, {vae_model.latent_dim-1}]"
    device = get_device(vae_model)

    if ref_img is not None:
        # Ensure ref_img is on the correct device and has batch dimension
        if ref_img.dim() == 3: # Assuming (C, H, W)
             ref_img = ref_img.unsqueeze(0) # Add batch dim -> (1, C, H, W)
        ref_img = ref_img.to(device)

        with torch.no_grad():
            # Encode the reference image
            # Assuming encoder returns dict with 'mean' and 'log_var' keys
            stats = vae_model.encoder(ref_img)['stats_qzx']
        
        # Unbind to get mean and log variance
        mu, logvar = stats.unbind(-1)

        # Use the encoded mean as the base latent vector
        base_latent = mu

        # Get mean for the specific index
        mean_val = mu[:, latent_factor_idx].item() 

        # Determine std deviation based on use_ref_img_lat_std
        if use_ref_img_lat_std:
            std_dev = torch.exp(0.5 * logvar[:, latent_factor_idx]).item() # Use std from encoder
        else:
            std_dev = 1.0 # Use default std

        # Get the traversal range based on the encoded mean and calculated/default std
        min_val, max_val = get_traversal_range(
            max_traversal_type=max_traversal_type, 
            max_traversal=max_traversal, 
            mean=mean_val, 
            std=std_dev
            )

    else:
        # Use the prior mean (0) as the base latent vector
        base_latent = torch.zeros(1, vae_model.latent_dim, device=device)
        # Get the traversal range based on the prior (mean=0, std=1)
        # When no ref_img, std is always 1, regardless of use_ref_img_lat_std
        min_val, max_val = get_traversal_range(max_traversal_type=max_traversal_type, 
                                               max_traversal=max_traversal, 
                                               mean=0, 
                                               std=1)

    # Repeat the base vector for the number of samples
    latent_vectors = base_latent.repeat(num_samples, 1)

    # Create the traversal values
    traversal_values = torch.linspace(min_val, max_val, num_samples, device=device)

    # Modify the specified latent dimension
    latent_vectors[:, latent_factor_idx] = traversal_values

    # Decode the latent vectors into images
    generated_images = decode_latents(vae_model, latent_vectors)
    return generated_images

def traverse_all_latents(vae_model,
                         num_samples=10,
                         max_traversal_type='probability', 
                         max_traversal=0.95, 
                         ref_img=None,
                         use_ref_img_lat_std=False # Add parameter
                         ):
    """
    Latent traversal for all latent dimensions.

    Parameters
    ----------
    vae_model : torch.nn.Module
        The VAE model.
    max_traversal_type : str
        Specifies how the traversal range is determined ('probability' or 'absolute').
    max_traversal : float
        The maximum traversal value, interpreted based on `max_traversal_type`.
    num_samples : int, optional
        The number of steps or images to generate along each traversal. Defaults to 10.
    ref_img : torch.Tensor, optional
        A reference image tensor (C, H, W) to base the traversal on. If None,
        traversal is based on the prior (mean=0, std=1). Defaults to None.
    use_ref_img_lat_std : bool, optional
        Passed to traverse_single_latent. If True and ref_img is provided,
        use the standard deviation from the encoder's variance output.
        Otherwise, use std=1. Defaults to False.

    Returns
    -------
    list of torch.Tensor
        A list containing tensors of generated images for each latent dimension.
        Each tensor has shape (num_samples, C, H, W).
    """
    all_traversals = []
    for latent_factor_idx in range(vae_model.latent_dim):
        # Pass ref_img and use_ref_img_lat_std to traverse_single_latent
        traversal_images = traverse_single_latent(vae_model=vae_model, 
                                                  latent_factor_idx=latent_factor_idx,
                                                  num_samples=num_samples,
                                                  max_traversal_type=max_traversal_type, 
                                                  max_traversal=max_traversal,  
                                                  ref_img=ref_img, 
                                                  use_ref_img_lat_std=use_ref_img_lat_std)
        all_traversals.append(traversal_images)
    return all_traversals