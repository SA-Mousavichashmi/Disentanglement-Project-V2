import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baseloss import BaseLoss

from ..reconstruction import reconstruction_loss
from .. import select


from .utils import Critic
# Import the moved functions
from .utils import  generate_latent_translations,\
                    apply_group_action_latent_space, \
                    select_latent_components, \
                    generate_latent_translations_selected_components

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
                 comp_latent_select_threshold=0,
                 base_loss_state_dict=None,
                 warm_up_steps=0,  # Add this parameter
                 **kwargs
                 ):
        
        super(Loss, self).__init__(mode="optimizes_internally",**kwargs)
        self.base_loss_name = base_loss_name # Base loss function for the model (like beta-vae, factor-vae, etc.)
        self.base_loss_kwargs = base_loss_kwargs # Base loss function kwargs
        self.base_loss_state_dict = base_loss_state_dict

        self.base_loss_f = select( 
                             name=self.base_loss_name, 
                             **self.base_loss_kwargs,
                             state_dict=self.base_loss_state_dict,
                             device=device
                             )  # base loss function

        self.rec_dist = rec_dist # for reconstruction loss type (especially for Identity loss)
        self.device = device
        self.deterministic_rep = deterministic_rep

        # Store the weights
        self.commutative_weight = commutative_weight
        self.meaningful_weight = meaningful_weight

        # if self.commutative_weight == 0 and self.meaningful_weight == 0:
        #     raise ValueError("At least one of commutative_weight or meaningful_weight must be greater than 0.")

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
        
        self.comp_latent_select_threshold = comp_latent_select_threshold
        if self.comp_latent_select_threshold < 0 or self.comp_latent_select_threshold >= 1:
            raise ValueError("comp_latent_select_threshold must be in the range [0, 1).")

        # Add warm-up parameters
        self.warm_up_steps = warm_up_steps
        self.current_step = 0

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
            'comp_latent_select_threshold': self.comp_latent_select_threshold,
            'warm_up_steps': self.warm_up_steps,
            'current_step': self.current_step
        }

    def state_dict(self):
        state = {}
        state['critic_state_dict'] = self.critic.state_dict() if self.critic is not None else None
        state['critic_optimizer_state_dict'] = self.critic_optimizer.state_dict() if self.critic_optimizer is not None else None
        state['base_loss_state_dict'] = self.base_loss_state_dict
        return state

    def load_state_dict(self, state_dict):
        if self.critic is not None:
            self.critic.load_state_dict(state_dict['critic_state_dict'])
        if self.critic_optimizer is not None:
            self.critic_optimizer.load_state_dict(state_dict['critic_optimizer_state_dict'])
        
        self.base_loss_state_dict = state_dict['base_loss_state_dict']
        self.base_loss_f.load_state_dict(self.base_loss_state_dict)
    
    def _group_action_commutative_loss(self, data, model, kl_components):
        """
        In this loss, we encourage that the group action in data space has commutative property:
        g.g'.x = g'.g.x for all g, g' in G and x in X.

        Args:
            data (torch.Tensor): Input data tensor of shape (batch_size, channels, height, width)
            model (nn.Module): VAE model containing encoder/decoder and required methods
            kl_components (torch.Tensor): KL divergence values per latent component (batch_size, latent_dim)

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
        _ , latent_dim = z.shape

        # 1: Select components randomly without replacement and its variances

        for comp_order in reversed(range(2, self.commutative_component_order + 1)):
            
            selected_row_indices, selected_component_indices = select_latent_components(
                component_order=comp_order,
                kl_components=kl_components,
                prob_threshold=self.comp_latent_select_threshold  # Changed parameter name
            )

            if selected_row_indices is not None and selected_component_indices is not None:
                break

        if selected_row_indices is None and selected_component_indices is None:
            raise ValueError(
                "No components selected based on the provided KL components and threshold. in _group_action_commutative_loss"
            )

        if len(selected_row_indices) != len(z):
            z = z[selected_row_indices]

        # Generate random transformation parameters using the selected components and their variances
        # Note: The function generate_latent_translations_selected_components is assumed to be defined elsewhere
        # and should generate random translations for the selected components based on their variances.

        g_component_index = selected_component_indices[:,0].unsqueeze(1)  # First component for g
        g_prime_component_indices = selected_component_indices[:, 1:]  # Remaining components for g'

        g = generate_latent_translations_selected_components(
            data_num=len(selected_row_indices),
            latent_dim=latent_dim,
            selected_components_indices=g_component_index,
        )
        gprime = generate_latent_translations_selected_components(
            data_num=len(selected_row_indices),
            latent_dim=latent_dim,
            selected_components_indices=g_prime_component_indices,
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
            commutative_loss = F.mse_loss(x_ggprime, x_gprimeg, reduction='sum') / data.size(0)  # Average over batch size
        elif self.rec_dist == 'bernoulli':
            pass # TODO: Implement it!

        return commutative_loss

    def _apply_multiple_group_actions_images(self, real_images, model):
        """
        Applies multiple consecutive group actions on input images through latent space transformations.
        This function performs a series of encode-transform-decode operations, where each iteration:
        1. Encodes the current image into latent space.
        2. Calculates KL components for the current latent representation.
        3. Selects latent components for group action based on these current KL components.
        4. Applies a random group action transformation in latent space using the selected components.
        5. Decodes back to image space.
        The output of each iteration becomes the input for the next.

        Args:
            real_images (torch.Tensor): Initial input images to transform.
            model: Model containing representation (encoder) and reconstruction (decoder) methods.

        Returns:
            torch.Tensor: Transformed images after applying multiple group actions.

        Note:
            The number of consecutive transformations is controlled by `self.meaningful_transformation_order`.
            The group action parameters are generated according to `self.meaningful_component_order`.
        """
        fake_images = real_images.clone()
        batch_size = real_images.size(0)
        # Get latent_dim from the second dimension of kl_components

        for _ in range(self.meaningful_transformation_order):
            # Encode
            # Note: We get z again here, but kl_components and variance_components passed correspond to the *original* image encoding
            z = model.get_representations(fake_images, is_deterministic=self.deterministic_rep)
            
            # Calculate kl_components for the current fake_images
            with torch.no_grad(): # Ensure no gradients are computed for KL components
                mean, logvar = model.encoder(fake_images)['stats_qzx'].unbind(-1)
                current_kl_components = kl_normal_loss(mean, logvar, raw=True)

            latent_dim = z.size(1)

            for comp_order in reversed(range(1, self.meaningful_component_order + 1)):
                
                selected_row_indices, selected_component_indices = select_latent_components(
                    component_order=comp_order,
                    kl_components=current_kl_components,
                    prob_threshold=self.comp_latent_select_threshold  # Changed parameter name
                )

                if selected_row_indices is not None and selected_component_indices is not None:
                    break

            if selected_row_indices is None and selected_component_indices is None:
                raise ValueError(
                    "No components selected based on the provided KL components and threshold. in group_action_meaningful_loss"
                )

            if len(selected_row_indices) != len(z):
                z = z[selected_row_indices]

            g = generate_latent_translations_selected_components(
                data_num=len(selected_row_indices),
                latent_dim=latent_dim,
                selected_components_indices=selected_component_indices,
            )
            
            # Apply group action in latent space using imported function
            z_transformed = apply_group_action_latent_space(g, z)
            # Decode => next "fake_images"
            fake_images = model.reconstruct_latents(z_transformed)

        return fake_images

    def __call__(self, data, model, vae_optimizer):
        log_data = {}
        base_loss = 0

        if self.base_loss_f.mode == 'post_forward':
            model_out = model(data)
            inputs = {
                'data': data,
                **model_out,
            }

            loss_out = self.base_loss_f(**inputs)
            base_loss = loss_out['loss']
            log_data.update(loss_out['to_log'])

        elif self.base_loss_f.mode == 'pre_forward':
            inputs = {
                'model': model,
                'data': data,
            }
            loss_out = self.base_loss_f(**inputs)
            base_loss = loss_out['loss']
            log_data.update(loss_out['to_log'])

        elif self.base_loss_f.mode == 'optimizes_internally':
            raise NotImplementedError("This loss function is not compatible with 'optimizes_internally' mode. for baseloss")

        with torch.no_grad():
            mean, logvar = model.encoder(data)['stats_qzx'].unbind(-1)

        kl_components_raw = kl_normal_loss(mean, logvar, raw=True) # shape: (batch_size, latent_dim)

        # Increment step counter
        self.current_step += 1
        
        # Group action losses (only after warm-up)
        group_loss = 0
        in_warm_up = self.current_step <= self.warm_up_steps
        
        # Commutative loss (skip during warm-up)
        if self.commutative_weight > 0 and not in_warm_up:
            g_commutative_loss = self._group_action_commutative_loss(data, model, kl_components_raw)
            log_data['g_commutative_loss'] = g_commutative_loss.item()
            group_loss += self.commutative_weight * g_commutative_loss

        # Meaningful loss (skip during warm-up)
        if self.meaningful_weight > 0 and not in_warm_up:
            
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
            )

            for p in self.critic.parameters(): p.requires_grad_(False)  # Freeze critic for generator update

            # WGAN generator loss (we want critic_fake to be high => negative sign)
            critic_fake = self.critic(fake_images)
            g_loss = -critic_fake.mean()
            log_data['generator_loss'] = g_loss.item()

            # Combine with the other group losses
            group_loss += self.meaningful_weight * g_loss
            total_loss = base_loss + group_loss  # plus any base VAE loss

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

