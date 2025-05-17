import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baseloss import BaseLoss

from ..reconstruction import reconstruction_loss
from .. import select


from .utils import Critic
# Import the moved functions
from .utils import generate_latent_translations, apply_group_action_latent_space, select_latent_components, generate_latent_translations_selected_components

from ..n_vae.kl_div import kl_normal_loss


class Loss(BaseLoss):
    """
    Compute Group theory losses in addition to base losses of models (BetaVAE, FactorVAE, etc.)

    The Group theory losses is as follows:
    1. Group action commutative loss
    2. Group action meaningful loss (Group equivariance)

    TODO: It is assumed that latent space is just consist of R^1 components to compatible to other models. But S^1 component with powerSpherical distribution should be added
    TODO: For Gaussian component instead instead of using fixed std for sampling from normal distribution, the std for each dim will learned approximately
    TODO: The discriminator must have the same architecture with Encoder
    """
    def __init__(self,
                 base_loss_name,
                 base_loss_kwargs,
                 rec_dist,
                 device,
                 commutative_weight,
                 commutative_component_order,
                 commutative_comparison_dist,
                 meaningful_weight,
                 meaningful_component_order,
                 meaningful_transformation_order,
                 meaningful_critic_gradient_penalty_weight,
                 meaningful_critic_lr,
                 meaningful_n_critic,
                 deterministic_rep,
                 **kwargs
                 ):
        
        super(Loss, self).__init__(**kwargs)
        self.base_loss_name = base_loss_name # Base loss function for the model (like beta-vae, factor-vae, etc.)
        self.base_loss_kwargs = base_loss_kwargs # Base loss function kwargs

        self.rec_dist = rec_dist # for reconstruction loss type (especially for Identity loss)
        self.device = device
        self.mode = 'optimizes_internally'
        self.deterministic_rep = deterministic_rep

        # Store the weights
        self.commutative_weight = commutative_weight
        self.meaningful_weight = meaningful_weight

        if self.commutative_weight == 0 and self.meaningful_weight == 0:
            raise ValueError("At least one of commutative_weight or meaningful_weight must be greater than 0.")

        # Store the component orders
        self.commutative_component_order = commutative_component_order
        self.meaningful_component_order = meaningful_component_order

        # Assert that commutative_component_order is greater than 1
        if self.commutative_component_order <= 1:
            raise ValueError("commutative_component_order must be greater than 1 for the commutative loss calculation.")
             
        # The parameters for the group
        self.meaningful_transformation_order = meaningful_transformation_order
        self.meaningful_critic_gradient_penalty_weight = meaningful_critic_gradient_penalty_weight
        self.meaningful_critic_lr = meaningful_critic_lr
        self.meaningful_n_critic = meaningful_n_critic

        # will be initialized in the first call to find the number of input channels
        self.critic = None
        self.critic_optimizer = None

        # Store commutative comparison distribution
        self.commutative_comparison_dist = commutative_comparison_dist
        if self.commutative_comparison_dist not in ['gaussian', 'bernoulli']:
            raise ValueError("commutative_comparison_dist must be either 'gaussian' or 'bernoulli'.")

    @property
    def name(self):
        return 'group_theory'

    @property
    def kwargs(self):
        return {
            'base_loss_name': self.base_loss_name,
            'base_loss_kwargs': self.base_loss_kwargs,
            'rec_dist': self.rec_dist,
            'device': self.device,
            'commutative_weight': self.commutative_weight,
            'commutative_component_order': self.commutative_component_order,
            'commutative_comparison_dist': self.commutative_comparison_dist,
            'meaningful_weight': self.meaningful_weight,
            'meaningful_component_order': self.meaningful_component_order,
            'meaningful_transformation_order': self.meaningful_transformation_order,
            'meaningful_critic_gradient_penalty_weight': self.meaningful_critic_gradient_penalty_weight,
            'meaningful_critic_lr': self.meaningful_critic_lr,
            'meaningful_n_critic': self.meaningful_n_critic,
            'deterministic_rep': self.deterministic_rep,
        }
    
    def _group_action_commutative_loss(self, data, model, kl_components, variance_components):
        """
        In this loss, we encourage that the group action in data space has commutative property:
        g.g'.x = g'.g.x for all g, g' in G and x in X.

        Args:
            data (torch.Tensor): Input data tensor of shape (batch_size, channels, height, width)
            model (nn.Module): VAE model containing encoder/decoder and required methods
            kl_components (torch.Tensor): KL divergence values per latent component (batch_size, latent_dim)
            variance_components (torch.Tensor): Variance values per latent component (batch_size, latent_dim)

        Returns:
            torch.Tensor: Commutative loss value calculated as MSE between x_gg' and x_g'g

        Note:
            Procedure steps:
            1. Get latent representation z from input data
            2. Select components for group actions using KL divergences
            3. Generate transformations g (single component) and g' (group components)
            4. Compute x_gg' through sequential g->g' transformations
            5. Compute x_g'g through sequential g'->g transformations
            6. Calculate MSE between the two transformed reconstructions
        """
        z = model.get_representations(data, is_deterministic=self.deterministic_rep)
        batch_size, latent_dim = z.shape

        # 1: Select components randomly without replacement and its variances
        selected_comp_indices = select_latent_components(
            self.commutative_component_order, kl_components
        )
        selected_variances = variance_components.gather(1, selected_comp_indices)


        # 2: select component index for g and g' based on the selected components
        # Note: The first selected component is the "single component group" (g), and the remaining are the "group component group" (g')
        g_comp_index = selected_comp_indices[:, 0].unsqueeze(1)  # First selected component (single component group)
        g_comp_variance = selected_variances[:, 0].unsqueeze(1)  # Variance of the first selected component

        gprime_comp_index = selected_comp_indices[:, 1:]  # Remaining selected components (group component group)
        gprime_comp_variance = selected_variances[:, 1:]  # Variances of the remaining selected components


        # Generate random transformation parameters using the selected components and their variances
        # Note: The function generate_latent_translations_selected_components is assumed to be defined elsewhere
        # and should generate random translations for the selected components based on their variances.

        g = generate_latent_translations_selected_components(
            batch_size=batch_size,
            latent_dim=latent_dim,
            selected_components_indices=g_comp_index,
            selected_components_variances=g_comp_variance
        )
        gprime = generate_latent_translations_selected_components(
            batch_size=batch_size,
            latent_dim=latent_dim,
            selected_components_indices=gprime_comp_index,
            selected_components_variances=gprime_comp_variance
        )

        # 5: Compute g.g'.x
        x_g = model.reconstruct_latents(apply_group_action_latent_space(g, z))
        z_g = model.get_representations(x_g, is_deterministic=self.deterministic_rep)
        x_ggprime = model.reconstruct_latents(apply_group_action_latent_space(gprime, z_g))

        # 6: Compute g'.g.x
        x_gprime = model.reconstruct_latents(apply_group_action_latent_space(gprime, z))
        z_gprime = model.get_representations(x_gprime, is_deterministic=self.deterministic_rep)
        x_gprimeg = model.reconstruct_latents(apply_group_action_latent_space(g, z_gprime))

        # 7: Compute comparison loss between x_gg' and x_g'g
        if self.commutative_comparison_dist == 'gaussian':
            commutative_loss = F.mse_loss(x_ggprime, x_gprimeg, reduction='mean')
        elif self.rec_dist == 'bernoulli':
            pass # TODO: Implement it!

        return commutative_loss

    def _apply_multiple_group_actions_images(self, real_images, model, kl_components, variance_components):
        """
        Applies multiple consecutive group actions on input images through latent space transformations.
        This function performs a series of encode-transform-decode operations, where each iteration:
        1. Encodes the image into latent space
        2. Applies a random group action transformation in latent space
        3. Decodes back to image space
        The output of each iteration becomes the input for the next.
        Args:
            real_images (torch.Tensor): Input images to transform
            model: Model containing representation (encoder) and reconstruction (decoder) methods
            kl_components (torch.Tensor): KL divergence per latent component.
            variance_components (torch.Tensor): Variance per latent component.
        Returns:
            torch.Tensor: Transformed images after applying multiple group actions
        Note:
            The number of consecutive transformations is controlled by self.meaningful_transformation_order
            The group action parameters are generated according to self.meaningful_component_order
        """
        fake_images = real_images.clone()
        batch_size = real_images.size(0)
        # Get latent_dim from the second dimension of kl_components
        latent_dim = kl_components.size(1)

        for _ in range(self.meaningful_transformation_order):
            # Encode
            # Note: We get z again here, but kl_components and variance_components passed correspond to the *original* image encoding
            z = model.get_representations(fake_images, is_deterministic=self.deterministic_rep)
            # Generate random transformation parameters using imported function
            transform_params = generate_latent_translations(
                batch_size=batch_size,
                latent_dim=latent_dim,
                component_order=self.meaningful_component_order,
                kl_components=kl_components, # Pass kl_components
                variance_components=variance_components # Pass variance_components
            )
            # Apply group action in latent space using imported function
            z_transformed = apply_group_action_latent_space(transform_params, z)
            # Decode => next "fake_images"
            fake_images = model.reconstruct_latents(z_transformed)

        return fake_images

    def __call__(self, data, model, vae_optimizer):
        is_train = model.training
        log_data = {}
        # TODO : Start from base model from specific state
        base_loss_f = select(device=data.device, name=self.base_loss_name, **self.base_loss_kwargs) # base loss function
        base_loss = 0

        if base_loss_f.mode == 'post_forward':
            model_out = model(data)
            inputs = {
                'data': data, 
                'is_train': False, 
                **model_out,
            }

            loss_out = base_loss_f(**inputs)
            base_loss = loss_out['loss']
            log_data.update(loss_out['to_log'])

        elif base_loss_f.mode == 'pre_forward':
            inputs = {
                'model': model,
                'data': data,
                'is_train': False
            }
            loss_out = base_loss_f(**inputs)
            base_loss = loss_out['loss']
            log_data.update(loss_out['to_log'])
        
        elif base_loss_f.mode == 'optimizes_internally':
            raise NotImplementedError("This loss function is not compatible with 'optimizes_internally' mode.")

        with torch.no_grad():
            mean, logvar = model.encoder(data)['stats_qzx'].unbind(-1)

        kl_components_raw = kl_normal_loss(mean, logvar, raw=True) # shape: (batch_size, latent_dim)
        variance_components = torch.exp(logvar) # Variance is exp(logvar) (batch_size, latent_dim)

        # Group action losses
        group_loss = 0

        # Commutative
        if self.commutative_weight > 0:
            g_commutative_loss = self._group_action_commutative_loss(data, model, kl_components_raw, variance_components)
            log_data['g_commutative_loss'] = g_commutative_loss.item()
            group_loss += self.commutative_weight * g_commutative_loss

        if self.meaningful_weight > 0:
            
            # Initialize critic if not already done based on the first data sample channel
            if  self.critic is None:
                input_channels_num = data.shape[1]
                self.critic = Critic(input_channels_num=input_channels_num).to(self.device)
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.meaningful_critic_lr)

            #################################################################
            # 1) Multiple critic (discriminator) updates first
            #################################################################
            d_losses = torch.zeros(self.meaningful_n_critic, device=data.device)
            for i in range(self.meaningful_n_critic):
                # Generate 'fake_images' without gradient for the critic
                with torch.no_grad():
                    fake_images = self._apply_multiple_group_actions_images(
                        real_images=data,
                        model=model,
                        kl_components=kl_components_raw, # Pass kl_components
                        variance_components=variance_components # Pass variance_components
                    )

                critic_real = self.critic(data)
                critic_fake = self.critic(fake_images)

                gp = self.critic._compute_gradient_penalty(data, fake_images)
                # WGAN-GP critic loss
                d_loss = -(critic_real.mean() - critic_fake.mean()) \
                         + self.meaningful_critic_gradient_penalty_weight * gp

                self.critic_optimizer.zero_grad()
                d_loss.backward()
                self.critic_optimizer.step()

                d_losses[i] = d_loss.item()

            # Log the average critic loss over the n_critic updates
            log_data['critic_loss'] = d_losses.mean().item()

            #################################################################
            # 2) Now update the generator (decoder) + group losses
            #################################################################
            # Re-generate fake images (this time with grad) for the generator update
            fake_images = self._apply_multiple_group_actions_images(
                real_images=data,
                model=model,
                kl_components=kl_components_raw, # Pass kl_components
                variance_components=variance_components # Pass variance_components
            )

            # WGAN generator loss (we want critic_fake to be high => negative sign)
            critic_fake = self.critic(fake_images)
            g_loss = -critic_fake.mean()
            log_data['generator_loss'] = g_loss.item()

            # Combine with the other group losses
            group_loss += self.meaningful_weight * g_loss
            total_loss = base_loss + group_loss  # plus any base VAE loss

            for p in self.critic.parameters(): p.requires_grad_(False)  # Freeze critic for generator update

            # Backprop through generator (i.e., the decoder) + group constraints
            vae_optimizer.zero_grad()
            total_loss.backward()
            vae_optimizer.step()

            for p in self.critic.parameters(): p.requires_grad_(True)  # Unfreeze critic

            #################################################################
            # 3) Final combined loss for logging
            #################################################################
            # Typically we just log generator and critic separately,
            # but here you can combine them if desired.
            final_loss = total_loss  # or total_loss + d_losses.mean() purely for logging
            log_data['loss'] = final_loss.item()

            return {'loss': final_loss, 'to_log': log_data}
        
        # Optimize group action losses only
        total_loss = base_loss + group_loss
        vae_optimizer.zero_grad()
        total_loss.backward()
        vae_optimizer.step()

        log_data['loss'] = total_loss.item()
        return {'loss': total_loss, 'to_log': log_data}

