import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baseloss import BaseLoss

from ..reconstruction import reconstruction_loss
from .. import select


class Critic(nn.Module):
    """
    Critic network for the WGAN-GP loss in the group meaningful loss.
    """
    def __init__(self, input_channels_num, activation_fn):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Conv2d(input_channels_num, 32, 4, stride=2, padding=1),  # 64x64xC -> 32x32x32
            activation_fn,
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # 32x32x32 -> 16x16x32
            activation_fn,
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 16x16x32 -> 8x8x64
            activation_fn,
            nn.Conv2d(64, 64, 4, stride=2, padding=1),  # 8x8x64 -> 4x4x64
            activation_fn,
            nn.Flatten(),
            nn.Linear(4 * 4 * 64, 256),
            activation_fn,
            nn.Linear(256, 1)  # Output scalar value
        )

    def forward(self, x):
        return self.critic(x)


def _compute_gradient_penalty(critic, real_images, fake_images):
    """
    Compute the gradient penalty for WGAN-GP.
    """
    device = real_images.device
    batch_size = real_images.size(0)

    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_images)

    interpolates = alpha * real_images + ((1 - alpha) * fake_images)
    interpolates.requires_grad_(True)

    disc_interpolates = critic(interpolates)

    grad_outputs = torch.ones_like(disc_interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


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
                 base_loss,
                 rec_dist,
                 device,
                 commutative_weight,
                 meaningful_weight,
                 commutative_component_order,
                 meaningful_component_order,
                 meaningful_transformation_order,
                 meaningful_critic_gradient_penalty_weight,
                 meaningful_critic_lr,
                 meaningful_n_critic,
                 deterministic_rep):
        super(Loss, self).__init__()
        self.base_loss = base_loss # Base loss function for the model (like beta-vae, factor-vae, etc.)
        self.rec_dist = rec_dist # for reconstruction loss type (especially for Identity loss)
        self.device = device
        self.mode = 'optimizes_internally'
        self.deterministic_rep = deterministic_rep

        # Store the weights
        self.commutative_weight = commutative_weight
        self.meaningful_weight = meaningful_weight

        # Store the component orders
        self.commutative_component_order = commutative_component_order
        self.meaningful_component_order = meaningful_component_order

        # The parameters for the group
        self.meaningful_transformation_order = meaningful_transformation_order
        self.meaningful_critic_gradient_penalty_weight = meaningful_critic_gradient_penalty_weight
        self.meaningful_critic_lr = meaningful_critic_lr
        self.meaningful_n_critic = meaningful_n_critic

        # will be initialized in the first call to find the number of input channels
        self.critic = None
        self.critic_optimizer = None
    
    def _generate_latent_transformation_random_parameters(self, batch_size, latent_dim, component_order):
        """
        Generate random parameters for latent transformation.
        """
        selected_components = torch.randperm(latent_dim)[:component_order]
        transformation_parameters = torch.randn(batch_size, latent_dim)

        mask = torch.zeros(latent_dim, dtype=torch.bool)
        mask[selected_components] = True
        transformation_parameters[:, ~mask] = 0  # set the components that will not be changed to 0

        return transformation_parameters


    def _apply_group_action_latent_space(self, transformation_parameters, latent_space):
        """
        Apply group action on latent space using transformation parameters.
        """
        transformed_latent_space = latent_space + transformation_parameters
        return transformed_latent_space

    def _group_action_commutative_loss(self, data, model):
        """
        In this loss, we encourage that the group action in data space has commutative property:
        g.g'.x = g'.g.x for all g, g' in G and x in X.

        The procedure is:
        1. Get the representation z of the input x (deterministic or stochastic).
        2. Select 'component_order' latent components randomly without replacement.
        3. Let the first selected component be the "single component group" (g), and the 
        remaining selected components be the "group component group" (g').
        4. Generate transformations for g and g' using generate_latent_transformation_random_parameters.
        5. Compute g.g'.x:
        - Apply g to z, decode to x_g.
        - Encode x_g to get z_g.
        - Apply g' to z_g, decode to x_gg'.
        6. Compute g'.g.x:
        - Apply g' to z, decode to x_g'.
        - Encode x_g' to get z_g'.
        - Apply g to z_g', decode to x_g'g.
        7. Compute the reconstruction loss between x_gg' and x_g'g to ensure commutation.
        """
        z = model.representation(data, deterministic=self.deterministic_rep)
        batch_size, latent_dim = z.shape

        # 2 & 3: Pick components for g (first) and g' (remaining)
        selected_components = torch.randperm(latent_dim)[:self.commutative_component_order]
        g_comp = selected_components[0]
        gprime_comp = selected_components[1:]

        # 4: Generate transformations for g and g'
        g = torch.zeros(batch_size, latent_dim, device=z.device)
        g[:, g_comp] = torch.randn(batch_size, device=z.device)
        
        gprime = torch.zeros(batch_size, latent_dim, device=z.device)
        gprime[:, gprime_comp] = torch.randn(batch_size, len(gprime_comp), device=z.device)

        # 5: Compute g.g'.x
        x_g = model.reconstruction(self._apply_group_action_latent_space(g, z))
        z_g = model.representation(x_g, deterministic=self.deterministic_rep)
        x_ggprime = model.reconstruction(self._apply_group_action_latent_space(gprime, z_g))

        # 6: Compute g'.g.x
        x_gprime = model.reconstruction(self._apply_group_action_latent_space(gprime, z))
        z_gprime = model.representation(x_gprime, deterministic=self.deterministic_rep)
        x_gprimeg = model.reconstruction(self._apply_group_action_latent_space(g, z_gprime))

        # 7: Compute reconstruction loss
        # Use the imported reconstruction_loss
        commutative_loss = reconstruction_loss(x_ggprime, x_gprimeg, distribution='gaussian') # TODO Check the correctness of this it is used mse loss
        return commutative_loss

    def _apply_multiple_group_actions_images(self, real_images, model):
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
        Returns:
            torch.Tensor: Transformed images after applying multiple group actions
        Note:
            The number of consecutive transformations is controlled by self.meaningful_transformation_order
            The group action parameters are generated according to self.meaningful_component_order
        """
        fake_images = real_images.clone()
        batch_size = real_images.size(0)
        
        for _ in range(self.meaningful_transformation_order):
            # Encode
            z = model.representation(fake_images, deterministic=self.deterministic_rep)
            # Generate random transformation parameters for your chosen latent dims
            transform_params = self._generate_latent_transformation_random_parameters(
                batch_size, 
                latent_dim=z.size(1),
                component_order=self.meaningful_component_order
            )
            # Apply group action in latent space
            z_transformed = self._apply_group_action_latent_space(transform_params, z)
            # Decode => next "fake_images"
            fake_images = model.reconstruction(z_transformed)
        
        return fake_images        

    def __call__(self, data, model, vae_optimizer, **kwargs):
        is_train = model.training
        self._pre_call(is_train)  # to match factor-vae style
        log_data = {}
        base_loss_f = select(device=data.device, name=self.base_loss) # base loss function
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


        # Group action losses
        group_loss=0

        # Identity loss removed
        log_data['g_identity_loss'] = 0

        # Compatibility loss removed
        log_data['g_compatibility_loss'] = 0

        # Commutative
        if self.commutative_weight > 0:
            g_commutative_loss = self._group_action_commutative_loss(data, model)
            log_data['g_commutative_loss'] = g_commutative_loss.item()
            group_loss += self.commutative_weight * g_commutative_loss
        else:
            log_data['g_commutative_loss'] = 0

        if self.meaningful_weight > 0:
            
            # Initialize critic if not already done based on the first data sample channel
            if  self.critic is None:
                input_channels_num = data.shape[1]
                self.critic = Critic(input_channels_num=input_channels_num, activation_fn=nn.LeakyReLU(0.2)).to(self.device)
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
                        model=model
                    )

                critic_real = self.critic(data)
                critic_fake = self.critic(fake_images)

                gp = _compute_gradient_penalty(self.critic, data, fake_images)
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
                model=model
            )

            # WGAN generator loss (we want critic_fake to be high => negative sign)
            critic_fake = self.critic(fake_images)
            g_loss = -critic_fake.mean()
            log_data['generator_loss'] = g_loss.item()

            # Combine with the other group losses
            total_group_loss = group_loss + self.meaningful_weight * g_loss
            total_loss = base_loss + total_group_loss  # plus any base VAE loss

            # Backprop through generator (i.e., the decoder) + group constraints
            vae_optimizer.zero_grad()
            total_loss.backward()
            vae_optimizer.step()

            #################################################################
            # 3) Final combined loss for logging
            #################################################################
            # Typically we just log generator and critic separately,
            # but here you can combine them if desired.
            final_loss = total_loss  # or total_loss + d_losses.mean() purely for logging
            log_data['loss'] = final_loss.item()

            return {'loss': final_loss, 'to_log': log_data}
        else:
            # Optimize group action losses only
            total_loss = base_loss + group_loss
            vae_optimizer.zero_grad()
            total_loss.backward()
            vae_optimizer.step()

            log_data['loss'] = total_loss.item()
            return {'loss': total_loss, 'to_log': log_data}
