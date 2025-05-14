import torch
from scipy import stats # Add scipy import
from vae_models import select

def get_device(model):
    """Gets the device of a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model.

    Returns
    -------
    torch.device
        The device the model is on.
    """
    return next(model.parameters()).device

def decode_latents(vae_model, latent_samples):
    """Decodes latent samples into images using the VAE's decoder.

    Parameters
    ----------
    vae_model : torch.nn.Module
        The VAE model.
    latent_samples : torch.Tensor
        A batch of latent vectors to be decoded. Shape (N, L), where N is the
        number of samples and L is the dimensionality of the latent space.

    Returns
    -------
    torch.Tensor
        The reconstructed images corresponding to the input latent samples.
        The tensor is moved to the CPU. Shape (N, C, H, W), where C, H, W are
        the image channels, height, and width.
    """
    device = get_device(vae_model)
    latent_samples = latent_samples.to(device)
    # Ensure decoder runs in inference mode
    with torch.no_grad():
        reconstructions = vae_model.decoder(latent_samples)['reconstructions']
    return reconstructions.cpu()

########################## Reconstruction Functions #########################

def reconstruct_ref_imgs(vae_model, ref_imgs, mode):
    """Reconstructs a list of reference images using the VAE model.

    Takes a list of image tensors, passes them through the VAE model to obtain
    reconstructions, and returns both the original images and their reconstructions.

    Parameters
    ----------
    vae_model : torch.nn.Module
        The VAE model used for reconstruction.
    ref_imgs : list of torch.Tensor
        A list containing the reference image tensors, each with shape (C, H, W).
    mode : str
        Mode for reconstruction. Options are 'mean' or 'sample'.

    Returns
    -------
    tuple of (list of torch.Tensor, list of torch.Tensor)
        A tuple containing two lists:
        - The first list holds the original reference images (on CPU).
        - The second list holds the corresponding reconstructed images generated
          by the VAE model (on CPU).
    """
    device = get_device(vae_model)
    # Stack images into a single tensor
    images = torch.stack(ref_imgs, dim=0)

    # Move images to the device
    images = images.to(device)

    # Reconstruct the images using the VAE model within no_grad context
    with torch.no_grad():
        reconstructions = vae_model.reconstruct(images, mode)

    # Move reconstructions back to CPU
    reconstructions = reconstructions.cpu()

    # Unbind tensors back into lists
    reconstructions_list = reconstructions.unbind(0)
    images_list = images.cpu().unbind(0) # Also move original images to CPU before unbinding

    return images_list, reconstructions_list

def reconstruct_sub_dataset(vae_model, dataset, img_indices, mode):
    """Reconstructs specific images from the dataset using the VAE model.

    Fetches images specified by their indices from the dataset, passes them
    through the VAE model to obtain reconstructions, and returns both the
    original images and their reconstructions.

    Parameters
    ----------
    vae_model : torch.nn.Module
        The VAE model used for reconstruction.
    dataset : torch.utils.data.Dataset
        The dataset from which images will be sampled.
    img_indices : list of int or torch.Tensor
        A list or tensor containing the indices of the images to reconstruct
        from the dataset.
    mode : str
        Mode for reconstruction. Options are 'mean' or 'sample'.

    Returns
    -------
    tuple of (list of torch.Tensor, list of torch.Tensor)
        A tuple containing two lists:
        - The first list holds the original images fetched from the dataset (on CPU).
        - The second list holds the corresponding reconstructed images generated
          by the VAE model (on CPU).
    """
    device = get_device(vae_model)
    # Get the images from the dataset
    images = [dataset[i][0] for i in img_indices]
    # Stack images into a single tensor
    images = torch.stack(images, dim=0)

    # Move images to the device
    images = images.to(device)

    # Reconstruct the images using the VAE model within no_grad context
    with torch.no_grad():
        reconstructions = vae_model.reconstruct(images, mode)

    # Move reconstructions back to CPU and return
    reconstructions = reconstructions.cpu()

    reconstructions_list = reconstructions.unbind(0)
    images_list = images.cpu().unbind(0) # Also move original images to CPU before unbinding

    return images_list, reconstructions_list

def random_reconstruct_sub_dataset(vae_model, dataset, num_samples=10, mode='mean'):
    """Reconstructs a specified number of randomly selected images from the dataset.

    Randomly selects indices from the dataset, retrieves the corresponding images,
    and uses the `reconstruct_sub_dataset` function to generate their VAE reconstructions.

    Parameters
    ----------
    vae_model : torch.nn.Module
        The VAE model used for reconstruction.
    dataset : torch.utils.data.Dataset
        The dataset from which images will be sampled.
    num_samples : int, optional
        The number of random images to select and reconstruct. Defaults to 10.
    mode : str, optional
        Mode for reconstruction. Options are 'mean' or 'sample'. Defaults to 'mean'.

    Returns
    -------
    tuple of (list of torch.Tensor, list of torch.Tensor)
        A tuple containing two lists, similar to the `reconstruct_sub_dataset` function:
        - The first list holds the randomly selected original images (on CPU).
        - The second list holds their corresponding reconstructions (on CPU).
    """
    # Get random indices from the dataset
    img_indices = torch.randint(0, len(dataset), (num_samples,))

    # Reconstruct the images using the VAE model
    images, reconstructions = reconstruct_sub_dataset(vae_model, dataset, img_indices, mode)

    return images, reconstructions

########## Load and Save Functions ##########

def create_load_model(model_name, model_kwargs, model_state_dict):
    model = select(model_name, **model_kwargs)
    model.load_state_dict(model_state_dict)
    return model
