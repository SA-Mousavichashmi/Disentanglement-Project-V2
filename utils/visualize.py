import matplotlib.pyplot as plt
from scipy import stats
import torch


class Visualizer():
    def __init__(self, vae_model, dataset, max_traversal_type='probability', max_traversal=0.475):
        """Initializes the visualizer class.

        Parameters
        ----------
        vae_model : torch.nn.Module
            The trained VAE model used for generating images and reconstructions.
        dataset : torch.utils.data.Dataset
            The dataset from which images will be sampled for visualization tasks.
        max_traversal_type : str, optional
            Specifies how the traversal range is determined. Must be either 'probability'
            or 'absolute'. Defaults to 'probability'.
            - 'probability': `max_traversal` defines the cumulative probability from the
              center of the Gaussian distribution (e.g., 0.45 means traversing from the
              5th percentile to the 95th percentile).
            - 'absolute': `max_traversal` defines the absolute range from the mean
              (e.g., 2 means traversing from mean - 2 to mean + 2).
        max_traversal : float, optional
            The maximum traversal value, interpreted based on `max_traversal_type`.
            Defaults to 0.475 for 'probability' type, corresponding to approximately
            +/- 1.96 standard deviations.
        """
        
        self.vae_model = vae_model # The vae model for generating images
        self.dataset = dataset # the dataset to be used for visualization
        self.device = next(self.vae_model.parameters()).device
        self.max_traversal_type = max_traversal_type
        self.max_traversal = max_traversal

        assert self.max_traversal_type in ['probability', 'absolute'], "max_traversal_type must be either 'probability' or 'absolute'."

        if self.max_traversal_type == 'probability':
            assert 0 <= self.max_traversal < 0.5, "max_traversal must be in [0, 0.5) for probability traversal."

        if self.max_traversal_type == 'absolute':
            assert self.max_traversal > 0, "max_traversal must be positive for absolute traversal."

    def _get_traversal_range(self, mean=0, std=1):
        """Calculates the absolute traversal range based on the specified type and value.

        This method determines the minimum and maximum values for latent variable traversals.
        The range is calculated based on the `max_traversal_type` and `max_traversal`
        attributes set during initialization. It assumes a standard normal distribution
        by default but can be adjusted with `mean` and `std` parameters.

        Parameters
        ----------
        mean : float, optional
            The mean of the latent variable distribution. Defaults to 0.
        std : float, optional
            The standard deviation of the latent variable distribution. Defaults to 1.

        Returns
        -------
        tuple of (float, float)
            A tuple containing the minimum and maximum traversal values in absolute terms.
            The range is symmetric around the mean.
        """

        if self.max_traversal_type == 'probability':
            # Calculate the z-score for the given probability
            z_score = stats.norm.ppf(self.max_traversal)
            # Convert z-score to absolute value using mean and std
            max_traversal = mean + (z_score * std)

        elif self.max_traversal_type == 'absolute':
            max_traversal = self.max_traversal

        # symmetrical traversals
        return (-1 * max_traversal, max_traversal)
    

    def _decode_latents(self, latent_samples):
        """Decodes latent samples into images using the VAE's decoder.

        Parameters
        ----------
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
        latent_samples = latent_samples.to(self.device)
        return self.model.decoder(latent_samples)['reconstructions'].cpu()
    
    ############# Latent Traversal Methods #############

    
    ############# Reconstruction Methods #############

    def reconstruct_ref_imgs(self, ref_imgs):
        """Reconstructs a list of reference images using the VAE model.

        Takes a list of image tensors, passes them through the VAE model to obtain
        reconstructions, and returns both the original images and their reconstructions.

        Parameters
        ----------
        ref_imgs : list of torch.Tensor
            A list containing the reference image tensors, each with shape (C, H, W).

        Returns
        -------
        tuple of (list of torch.Tensor, list of torch.Tensor)
            A tuple containing two lists:
            - The first list holds the original reference images (on CPU).
            - The second list holds the corresponding reconstructed images generated
              by the VAE model (on CPU).
        """
        # Stack images into a single tensor
        images = torch.stack(ref_imgs, dim=0) 
        
        # Move images to the device
        images = images.to(self.device)
        
        # Reconstruct the images using the VAE model within no_grad context
        with torch.no_grad():
            reconstructions = self.vae_model(images)['reconstructions']
        
        # Move reconstructions back to CPU
        reconstructions = reconstructions.cpu()

        # Unbind tensors back into lists
        reconstructions_list = reconstructions.unbind(0)

        return reconstructions_list

    def reconstruct_sub_dataset(self, img_indices):
        """Reconstructs specific images from the dataset using the VAE model.

        Fetches images specified by their indices from the dataset, passes them
        through the VAE model to obtain reconstructions, and returns both the
        original images and their reconstructions.

        Parameters
        ----------
        img_indices : list of int or torch.Tensor
            A list or tensor containing the indices of the images to reconstruct
            from the dataset.

        Returns
        -------
        tuple of (list of torch.Tensor, list of torch.Tensor)
            A tuple containing two lists:
            - The first list holds the original images fetched from the dataset (on CPU).
            - The second list holds the corresponding reconstructed images generated
              by the VAE model (on CPU).
        """
        # Get the images from the dataset
        images = [self.dataset[i][0] for i in img_indices]
        # Stack images into a single tensor
        images = torch.stack(images, dim=0) 
        
        # Move images to the device
        images = images.to(self.device)
        
        # Reconstruct the images using the VAE model within no_grad context
        with torch.no_grad():
            reconstructions = self.vae_model(images)['reconstructions']
        
        # Move reconstructions back to CPU and return
        reconstructions = reconstructions.cpu()

        reconstructions = reconstructions.unbind(0)
        images = images.cpu().unbind(0) # Also move original images to CPU before unbinding

        return images, reconstructions
    
    def random_reconstruct_sub_dataset(self, num_samples=10):
        """Reconstructs a specified number of randomly selected images from the dataset.

        Randomly selects indices from the dataset, retrieves the corresponding images,
        and uses the `reconstruct` method to generate their VAE reconstructions.

        Parameters
        ----------
        num_samples : int, optional
            The number of random images to select and reconstruct. Defaults to 10.

        Returns
        -------
        tuple of (list of torch.Tensor, list of torch.Tensor)
            A tuple containing two lists, similar to the `reconstruct` method:
            - The first list holds the randomly selected original images (on CPU).
            - The second list holds their corresponding reconstructions (on CPU).
        """
        # Get random indices from the dataset
        img_indices = torch.randint(0, len(self.dataset), (num_samples,))
        
        # Reconstruct the images using the VAE model
        images, reconstructions = self.reconstruct_sub_dataset(img_indices)
        
        return images, reconstructions
    
    def plot_reconstructions(self, imgs, reconstructions, figsize=(10, 3)):
        """Plots original images and their reconstructions side-by-side.

        Creates a matplotlib figure displaying the original images in the top row
        and their corresponding reconstructions in the bottom row.

        Parameters
        ----------
        imgs : list of torch.Tensor
            A list of original image tensors (on CPU).
        reconstructions : list of torch.Tensor
            A list of reconstructed image tensors (on CPU).
        figsize : tuple, optional
            The size of the matplotlib figure. Defaults to (10, 3).
        """
        num_images = len(imgs)
        fig, axes = plt.subplots(2, num_images, figsize=figsize)

        # Handle case where num_images is 1, axes is not a 2D array
        if num_images == 1:
            axes = axes.reshape(2, 1)

        for i in range(num_images):
            # Plot original image
            ax = axes[0, i]
            img = imgs[i].permute(1, 2, 0).numpy() # Convert CHW to HWC for plotting
            ax.imshow(img)
            ax.axis('off')
            if i == 0:
                ax.set_title('Original', fontsize=10)

            # Plot reconstructed image
            ax = axes[1, i]
            recon = reconstructions[i].permute(1, 2, 0).numpy() # Convert CHW to HWC
            ax.imshow(recon)
            ax.axis('off')
            if i == 0:
                ax.set_title('Reconstruction', fontsize=10)

        plt.tight_layout(pad=0.1)
        plt.show()
    
    def plot_random_reconstructions(self, num_samples=10, figsize=(10, 3)):
        """Randomly selects and plots a specified number of images and their reconstructions.

        This method combines the functionality of `random_reconstruct_sub_dataset` and
        `plot_reconstructions` to display a set of randomly chosen images alongside
        their VAE reconstructions.

        Parameters
        ----------
        num_samples : int, optional
            The number of random images to select and reconstruct. Defaults to 10.
        figsize : tuple, optional
            The size of the matplotlib figure. Defaults to (10, 3).
        """
        imgs, reconstructions = self.random_reconstruct_sub_dataset(num_samples)
        self.plot_reconstructions(imgs, reconstructions, figsize)
    
    def plot_reconstructions_sub_dataset(self, img_indices, figsize=(10, 3)):
        """Reconstructs and plots images from the dataset specified by their indices.

        This method combines the functionality of `reconstruct_sub_dataset` and
        `plot_reconstructions` to display a set of images alongside their VAE
        reconstructions.

        Parameters
        ----------
        img_indices : list of int or torch.Tensor
            A list or tensor containing the indices of the images to reconstruct
            from the dataset.
        figsize : tuple, optional
            The size of the matplotlib figure. Defaults to (10, 3).
        """
        imgs, reconstructions = self.reconstruct_sub_dataset(img_indices)
        self.plot_reconstructions(imgs, reconstructions, figsize)





