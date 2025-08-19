import torch
import torch.nn.functional as F
from collections import OrderedDict

from ..baseloss import BaseLoss
from .. import select

from .utils import  apply_group_action_latent_space, \
                    select_latent_components, \
                    generate_latent_translations_selected_components

from .gan.trainer import GANTrainer
from ..n_vae.kl_div import kl_normal_loss


class Loss(BaseLoss):
    """    Compute Group theory losses in addition to base losses of models (BetaVAE, FactorVAE, etc.)
    """
    
    def __init__(self,
                 base_loss_name,
                 base_loss_kwargs,
                 rec_dist,
                 device,
                 img_size,
                 # g-commutative loss parameters
                 commutative_weight=1,
                 commutative_component_order=2,
                 commutative_comparison_dist= 'gaussian',
                 # g-meaningful loss parameters
                 meaningful_weight=1,
                 meaningful_component_order=2,
                 meaningful_transformation_order=1,
                 meaningful_n_critic=1,
                 meaningful_gan_config=None,
                # general group theory parameters
                 deterministic_rep=True,
                 group_action_latent_range=2,
                 group_action_latent_distribution='uniform',
                 comp_latent_select_threshold=0,
                 meaningful_critic_betas=(0.9, 0.999),
                 meaningful_critic_eps=1e-8,
                 meaningful_critic_weight_decay=0.0,
                 warm_up_steps=0,  # Add this parameter
                 # schedulers kwargs
                 schedulers_kwargs=None,
                 **kwargs
                 ):
        
        ##### parameters that is compatible for scheduling #####
        self.commutative_weight = commutative_weight
        self.meaningful_weight = meaningful_weight

        super(Loss, self).__init__(mode="optimizes_internally", 
                                   rec_dist=rec_dist, 
                                   schedulers_kwargs=schedulers_kwargs, 
                                   **kwargs)
        
        if self.schedulers: 
            # Validate scheduler configuration
            allowed_schedulers = {'commutative_weight', 'meaningful_weight'}
            provided_schedulers = set(self.schedulers.keys())
            
            if not provided_schedulers.issubset(allowed_schedulers):
                invalid_schedulers = provided_schedulers - allowed_schedulers
                raise ValueError(
                    f"Invalid scheduler configuration. Group Theory loss only supports schedulers for {list(allowed_schedulers)}, "
                    f"but found unexpected schedulers for: {list(invalid_schedulers)}"
                )
            
            if 'commutative_weight' in self.schedulers:
                self.commutative_weight = self.schedulers['commutative_weight'].get_value()
            if 'meaningful_weight' in self.schedulers:
                self.meaningful_weight = self.schedulers['meaningful_weight'].get_value()

        self.base_loss_name = base_loss_name # Base loss function for the model (like beta-vae, factor-vae, etc.)
        self.base_loss_kwargs = base_loss_kwargs # Base loss function kwargs

        self.base_loss_f = select( 
                             name=self.base_loss_name, 
                             **self.base_loss_kwargs,
                             device=device
                             )  # base loss function
        
        # If using FactorVAE with group theory, ensure external optimization is enabled
        if self.base_loss_name == 'factor_vae' and not self.base_loss_f.external_optimization:
            raise ValueError(
                "When using FactorVAE with group theory loss, the FactorVAE must be configured "
                "with external_optimization=True to allow proper discriminator update scheduling."
            )

        self.rec_dist = rec_dist # for reconstruction loss type (especially for Identity loss)
        self.device = device
        self.img_size = img_size  # Store image size (channels, height, width)
        self.deterministic_rep = deterministic_rep        # Store the weights
        self.group_action_latent_range = group_action_latent_range
        self.group_action_latent_distribution = group_action_latent_distribution

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
        self.meaningful_n_critic = meaningful_n_critic

        # Store GAN configuration
        self.meaningful_gan_config = meaningful_gan_config

        # Initialize GAN trainer immediately if meaningful_weight > 0
        self.gan_trainer = None
        if self.meaningful_gan_config is not None:
            self.gan_trainer = GANTrainer(
                img_size=self.img_size,
                device=self.device,
                **self.meaningful_gan_config
            )

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

        self.latent_factor_topologies = None

    @property
    def name(self):
        return 'group_theory'
    
    @property
    def kwargs(self):
        kwargs_dict = {
            'base_loss_name': self.base_loss_name,
            'base_loss_kwargs': self.base_loss_kwargs,
            'rec_dist': self.rec_dist,
            'device': self.device,
            'img_size': self.img_size,
            'warm_up_steps': self.warm_up_steps,
            'current_step': self.current_step,
            # g-commutative loss parameters
            'commutative_weight': self.commutative_weight,
            'commutative_component_order': self.commutative_component_order,
            'commutative_comparison_dist': self.commutative_comparison_dist,
            # g-meaningful loss parameters
            'meaningful_weight': self.meaningful_weight,
            'meaningful_component_order': self.meaningful_component_order,
            'meaningful_transformation_order': self.meaningful_transformation_order,
            'meaningful_n_critic': self.meaningful_n_critic,
            'meaningful_gan_config': self.meaningful_gan_config,
            # general group theory parameters
            'deterministic_rep': self.deterministic_rep,
            'group_action_latent_range': self.group_action_latent_range,
            'group_action_latent_distribution': self.group_action_latent_distribution,
            'comp_latent_select_threshold': self.comp_latent_select_threshold,
        }
        
        # Add scheduler configurations
        if self.schedulers:
            schedulers_kwargs = []
            for param_name, scheduler in self.schedulers.items():
                schedulers_kwargs.append({
                    'name': scheduler.name,
                    'param_name': param_name,
                    'kwargs':{**scheduler.kwargs}
                })
                
            kwargs_dict['schedulers_kwargs'] = schedulers_kwargs

        return kwargs_dict

    def state_dict(self):
        state = {}
        state['gan_trainer_state_dict'] = self.gan_trainer.state_dict() if self.gan_trainer is not None else None
        state['base_loss_state_dict'] = self.base_loss_state_dict
        
        # Save scheduler states
        if self.schedulers:
            state['scheduler_states'] = {}
            for param_name, scheduler in self.schedulers.items():
                state['scheduler_states'][param_name] = scheduler.state_dict()
                
        return state

    def load_state_dict(self, state_dict):
        if self.gan_trainer is not None and state_dict.get('gan_trainer_state_dict') is not None:
            self.gan_trainer.load_state_dict(state_dict['gan_trainer_state_dict'])
        
        self.base_loss_state_dict = state_dict['base_loss_state_dict']
        self.base_loss_f.load_state_dict(self.base_loss_state_dict)
        
        # Load scheduler states
        if 'scheduler_states' in state_dict and self.schedulers:
            for param_name, scheduler_state in state_dict['scheduler_states'].items():
                if param_name in self.schedulers:
                    self.schedulers[param_name].load_state_dict(scheduler_state)

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
            range=self.group_action_latent_range,
            distribution=self.group_action_latent_distribution,
        )
        gprime = generate_latent_translations_selected_components(
            data_num=len(selected_row_indices),
            latent_dim=latent_dim,
            selected_components_indices=g_prime_component_indices,
            range=self.group_action_latent_range,
            distribution=self.group_action_latent_distribution,
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
            range=self.group_action_latent_range,
            distribution=self.group_action_latent_distribution,
        )
        
        # Apply group action in latent space using imported function
        z_transformed = apply_group_action_latent_space(g, z)
        # Decode => next "fake_images"
        fake_images = model.reconstruct_latents(z_transformed)

        return fake_images

    def __call__(self, data, model, vae_optimizer):

        if self.latent_factor_topologies is None:
            self.latent_factor_topologies = model.latent_factor_topologies

        log_data = OrderedDict()
        base_loss = 0

        if self.base_loss_f.mode == 'post_forward':
            model_out = model(data)
            inputs = {
                'data': data,
                **model_out,
                'is_train': True,
            }

            loss_out = self.base_loss_f(**inputs)
            base_loss = loss_out['loss']
            log_data.update(loss_out['to_log'])

        elif self.base_loss_f.mode == 'pre_forward':
            inputs = {
                'model': model,
                'data': data,
                'is_train': True,
            }
            loss_out = self.base_loss_f(**inputs)
            base_loss = loss_out['loss']
            log_data.update(loss_out['to_log'])

        elif self.base_loss_f.mode == 'optimizes_internally':
            raise NotImplementedError(
                "The base loss function is set to 'optimizes_internally', "
                "but this mode is not supported in the group theory loss. "
                "Please use 'post_forward' or 'pre_forward' modes."
            )

        if self.base_loss_f.name == 'factorvae':
             _, data_Bp = torch.chunk(data, 2, dim=0)

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

                d_loss = self.gan_trainer.train_discriminator(data, fake_images, weight=self.meaningful_weight)
                d_losses[i] = d_loss

            # Log the average critic loss over the n_critic updates
            log_data['g_meaningful_critic'] = d_losses.mean().item()

            #################################################################
            # 2) Now update the generator (decoder) + group losses
            #################################################################
            # Re-generate fake images (this time with grad) for the generator update
            fake_images = self._apply_multiple_group_actions_images(
                real_images=data,
                model=model,
            )

            # Freeze discriminator for generator update
            self.gan_trainer.freeze_discriminator()

            # Compute generator loss using the GAN trainer
            g_loss = self.gan_trainer.compute_generator_loss(fake_images)
            log_data['g_meaningful_generator'] = g_loss.item()

            # Combine with the other group losses
            total_loss = base_loss + self.meaningful_weight * g_loss  # plus any base VAE loss

            # Backprop through generator (i.e., the decoder) + group constraints
            vae_optimizer.zero_grad()
            total_loss.backward()
            vae_optimizer.step()

            # Unfreeze discriminator
            self.gan_trainer.unfreeze_discriminator()

            if self.base_loss_f.name == 'factorvae':
                # Update FactorVAE discriminator after VAE optimization
                discr_result = self.base_loss_f.update_discriminator(data_Bp, model)
                log_data.update(discr_result['to_log'])

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

        if self.base_loss_f.name == 'factorvae':
            # Update FactorVAE discriminator after VAE optimization
            discr_result = self.base_loss_f.update_discriminator(data_Bp, model)
            log_data.update(discr_result['to_log'])

        log_data['loss'] = total_loss.item()
        return {'loss': total_loss, 'to_log': log_data}

