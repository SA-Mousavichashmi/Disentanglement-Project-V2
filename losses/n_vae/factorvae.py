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

    # ------------ main loss call -------------------------------------------------
    def __call__(self, data, model, optimizer=None, **kwargs):
        """
        One full optimization step or loss computation:
            - If external_optimization=False: update encoder/decoder (θ) + discriminator (ψ)
            - If external_optimization=True: only update discriminator (ψ), return VAE loss for external optimization
        """
        log_dict = {}
        batch_size = data.size(0)

        # ------------------------------------------------------------------ #
        #  STEP A – VAE loss computation                                      #
        # ------------------------------------------------------------------ #
        data_B, data_Bp = torch.chunk(data, 2, dim=0)   # B  and B′ (equal halves)

        # forward B through VAE
        out = model(data_B)
        if isinstance(out["stats_qzx"], torch.Tensor):
            out["stats_qzx"] = out["stats_qzx"].unbind(-1)

        rec_loss = reconstruction_loss(
            data_B, out["reconstructions"], distribution=self.rec_dist
        )
        kl_comp = kl_normal_loss(*out["stats_qzx"], return_components=True)
        kl_loss = kl_comp.sum()

        # total-correlation estimator (log D₀ − log D₁)
        for p in self.discriminator.parameters(): p.requires_grad_(False)

        tc_logits = self.discriminator(out["samples_qzx"])
        tc_loss = (tc_logits[:, 0] - tc_logits[:, 1]).mean()

        vae_loss = rec_loss + kl_loss + self.gamma * tc_loss

        # ---- VAE optimization (only if not using external optimizer)
        if not self.external_optimization:
            if optimizer is None:
                raise ValueError("optimizer must be provided when external_optimization=False")
            optimizer.zero_grad()
            vae_loss.backward()
            optimizer.step()
            # After optimization, detach for discriminator step
            vae_loss_for_log = vae_loss.detach()
        else:
            # Keep gradients for external optimization
            vae_loss_for_log = vae_loss

        log_dict.update(
            {
                "loss": vae_loss_for_log.item() if torch.is_tensor(vae_loss_for_log) else vae_loss_for_log,
                "rec_loss": rec_loss.item(),
                "kl_loss": kl_loss.item(),
                "tc_loss": tc_loss.item(),
            }
        )
        if self.log_kl_components:
            log_dict.update(
                {f"kl_loss_{i}": v.item() for i, v in enumerate(kl_comp)}
            )

        # ------------------------------------------------------------------ #
        #  STEP B – discriminator update (ψ) - always performed              #
        # ------------------------------------------------------------------ #
        for p in self.discriminator.parameters(): p.requires_grad_(True)

        # latent batch z′ from B′ *after* θ has just been updated (or not)
        with torch.no_grad():
            z_real = model.sample_qzx(data_Bp).detach()   # no grad to θ

        z_perm = _permute_dims(z_real)

        logits_real = self.discriminator(z_real)          # grad → ψ only
        logits_perm = self.discriminator(z_perm)

        target_real = torch.zeros(len(z_real), dtype=torch.long, device=self.device)
        target_perm = torch.ones_like(target_real)

        discr_loss = 0.5 * (
            F.cross_entropy(logits_real, target_real)
            + F.cross_entropy(logits_perm, target_perm)
        )

        self.optimizer_d.zero_grad()
        discr_loss.backward()
        self.optimizer_d.step()

        log_dict["discrim_loss"] = discr_loss.item()

        # Return based on mode
        if self.external_optimization:
            # Return VAE loss for external optimization
            return {"loss": vae_loss, "to_log": log_dict}
        else:
            # Return detached loss (already optimized internally)
            return {"loss": vae_loss.detach(), "to_log": log_dict}

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
        self.leaky_relu = nn.LeakyReLU(self.neg_slope, False)

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

        # Fully connected layers with leaky ReLu activations
        z = self.leaky_relu(self.lin1(z))
        z = self.leaky_relu(self.lin2(z))
        z = self.leaky_relu(self.lin3(z))
        z = self.leaky_relu(self.lin4(z))
        z = self.leaky_relu(self.lin5(z))
        z = self.leaky_relu(self.lin6(z))
        z = self.lin7(z)

        return z

    def reset_parameters(self):
        self.apply(initialization.weights_init)
