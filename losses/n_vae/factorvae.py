# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import baseloss
from utils import initialization
from ..reconstruction import reconstruction_loss
from .kl_div import kl_normal_loss
from .utils import _permute_dims

class Loss(baseloss.BaseLoss):
    """
    FactorVAE loss faithful to Algorithm 2 (Kim & Mnih, 2018).

    Parameters
    ----------
    device      : torch.device
    gamma       : float      weight for TC term
    discr_lr    : float      ψ learning-rate
    discr_betas : tuple      Adam β parameters
    external_optimization : bool
        If True, the VAE loss is returned without internal optimization (mode='post_forward').
        If False, the VAE loss is optimized internally as usual (mode='optimizes_internally').
        Default: False (maintains backward compatibility).
        
        When True, the discriminator is still updated internally, but the VAE loss 
        can be combined with other losses (e.g., group theory losses) for external optimization.
    """
    def __init__(
        self,
        device,
        gamma=6.4,
        discr_lr=5e-5,
        discr_betas=(0.5, 0.9),
        log_kl_components=False,
        external_optimization=False,
        **kwargs,
    ):
        # Set mode based on whether external optimization is used
        mode = "pre_forward" if external_optimization else "optimizes_internally"
        super().__init__(mode=mode, **kwargs)
        self.device = device
        self.gamma = gamma
        self.external_optimization = external_optimization
        self.discriminator = FactorDiscriminator().to(device)
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=discr_lr, betas=discr_betas
        )
        self.log_kl_components = log_kl_components

    # ------------ public helpers ------------------------------------------------
    @property
    def name(self):
        return "factorvae"

    @property
    def kwargs(self):
        return {
            "gamma": self.gamma,
            "discr_lr": self.optimizer_d.param_groups[0]["lr"],
            "discr_betas": self.optimizer_d.param_groups[0]["betas"],
            "log_kl_components": self.log_kl_components,
            "external_optimization": self.external_optimization,
            "rec_dist": self.rec_dist,
        }

    # ------------ separate methods for VAE loss and discriminator update --------
    def compute_vae_loss(self, data_B, model):
        """
        Compute the FactorVAE loss without optimization.
        
        Args:
            data: Input data tensor
            model: VAE model
            
        Returns:
            dict: Contains 'loss', 'components' (individual loss terms), and 'to_log' (logging dict)
        """
        log_dict = {}
        
        # Forward pass through VAE
        out = model(data_B)
        if isinstance(out["stats_qzx"], torch.Tensor):
            out["stats_qzx"] = out["stats_qzx"].unbind(-1)
        
        # Compute loss components
        rec_loss = reconstruction_loss(
            data_B, out["reconstructions"], distribution=self.rec_dist
        )
        kl_comp = kl_normal_loss(*out["stats_qzx"], return_components=True)
        kl_loss = kl_comp.sum()
        
        # Total-correlation estimator (log D₀ − log D₁)
        # Freeze discriminator parameters for VAE loss computation
        for p in self.discriminator.parameters(): 
            p.requires_grad_(False)
        
        tc_logits = self.discriminator(out["samples_qzx"])
        tc_loss = (tc_logits[:, 0] - tc_logits[:, 1]).mean()
        
        # Restore discriminator gradients
        # for p in self.discriminator.parameters(): 
        #     p.requires_grad_(True)
        
        # Total VAE loss
        vae_loss = rec_loss + kl_loss + self.gamma * tc_loss
        
        # Prepare logging
        log_dict.update({
            "loss": vae_loss.item(),
            "rec_loss": rec_loss.item(),
            "kl_loss": kl_loss.item(),
            "tc_loss": tc_loss.item(),
        })
        
        if self.log_kl_components:
            log_dict.update({f"kl_loss_{i}": v.item() for i, v in enumerate(kl_comp)})
        
        return {
            "loss": vae_loss,
            "to_log": log_dict
        }
    
    def update_discriminator(self, data_Bp, model):
        """
        Update the discriminator using the second half of the data.
        
        Args:
            data_Bp: Second half of the data batch for discriminator update
            model: VAE model
            
        Returns:
            dict: Contains discriminator loss and logging information
        """
        # Ensure discriminator gradients are enabled
        for p in self.discriminator.parameters(): 
            p.requires_grad_(True)
        
        # Generate latent samples from the second half of data
        with torch.no_grad():
            z_real = model.sample_qzx(data_Bp)   # no grad to θ
        
        z_perm = _permute_dims(z_real)
        
        # Discriminator forward pass
        logits_real = self.discriminator(z_real)
        logits_perm = self.discriminator(z_perm)
        
        # Targets for classification
        target_real = torch.zeros(len(z_real), dtype=torch.long, device=self.device)
        target_perm = torch.ones_like(target_real)
        
        # Discriminator loss
        discr_loss = 0.5 * (
            F.cross_entropy(logits_real, target_real)
            + F.cross_entropy(logits_perm, target_perm)
        )
        
        # Update discriminator
        self.optimizer_d.zero_grad()
        discr_loss.backward()
        self.optimizer_d.step()
        
        return {
            "discrim_loss": discr_loss.item(),
            "to_log": {"discrim_loss": discr_loss.item()}
        }
    
    # ------------ main loss call -------------------------------------------------
    def __call__(self, data, model, optimizer=None, **kwargs):
        """
        One full optimization step or loss computation:
            - If external_optimization=False: update encoder/decoder (θ) + discriminator (ψ)
            - If external_optimization=True: return VAE loss for external optimization, skip discriminator update
        """

        # Split data for VAE loss computation
        data_B, data_Bp = torch.chunk(data, 2, dim=0)   # B and B′ (equal halves)
        
        # Compute VAE loss
        vae_result = self.compute_vae_loss(data_B, model)
        vae_loss = vae_result["loss"]
        log_dict = vae_result["to_log"]
        
        if not self.external_optimization:
            # Internal optimization mode: optimize VAE and discriminator
            if optimizer is None:
                raise ValueError("optimizer must be provided when external_optimization=False")
            
            # Optimize VAE
            optimizer.zero_grad()
            vae_loss.backward()
            optimizer.step()
            
            # Update discriminator
            discr_result = self.update_discriminator(data_Bp, model)
            log_dict.update(discr_result["to_log"])
            
            return {"loss": vae_loss.detach(), "to_log": log_dict}
        else:
            # External optimization mode: return VAE loss with gradients for external optimizer
            # Do not update discriminator here - it will be updated externally
            return {"loss": vae_loss, "to_log": log_dict}

    # ------------ checkpoint helpers -------------------------------------------
    def state_dict(self):
        return {
            "discriminator": self.discriminator.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
        }

    def load_state_dict(self, state):
        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer_d.load_state_dict(state["optimizer_d"])

class FactorDiscriminator(nn.Module):

    def __init__(self, neg_slope=0.2, latent_dim=10, hidden_units=1000):
        """Discriminator proposed in [1].

        Parameters
        ----------
        neg_slope: float
            Hyperparameter for the Leaky ReLu

        latent_dim : int
            Dimensionality of latent variables.

        hidden_units: int
            Number of hidden units in the MLP

        Model Architecture
        ------------
        - 6 layer multi-layer perceptron, each with 1000 hidden units
        - Leaky ReLu activations
        - Output 2 logits

        References:
            [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorizing."
            arXiv preprint arXiv:1802.05983 (2018).

        """
        super(FactorDiscriminator, self).__init__()

        # Activation parameters
        self.neg_slope = neg_slope
        self.leaky_relu = nn.LeakyReLU(self.neg_slope, True)

        # Layer parameters
        self.z_dim = latent_dim
        self.hidden_units = hidden_units
        # theoretically 1 with sigmoid but gives bad results => use 2 and softmax
        out_units = 2

        # Fully connected layers
        self.lin1 = nn.Linear(self.z_dim, hidden_units)
        self.lin2 = nn.Linear(hidden_units, hidden_units)
        self.lin3 = nn.Linear(hidden_units, hidden_units)
        self.lin4 = nn.Linear(hidden_units, hidden_units)
        self.lin5 = nn.Linear(hidden_units, hidden_units)
        self.lin6 = nn.Linear(hidden_units, hidden_units)
        self.lin7 = nn.Linear(hidden_units, out_units)

        self.reset_parameters()

    def forward(self, z):
        # Use intermediate variables instead of reusing z
        h = self.leaky_relu(self.lin1(z))
        h = self.leaky_relu(self.lin2(h))
        h = self.leaky_relu(self.lin3(h))
        h = self.leaky_relu(self.lin4(h))
        h = self.leaky_relu(self.lin5(h))
        h = self.leaky_relu(self.lin6(h))
        out = self.lin7(h)
        return out

    def reset_parameters(self):
        self.apply(initialization.weights_init)
