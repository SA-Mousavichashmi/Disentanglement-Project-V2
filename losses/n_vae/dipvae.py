# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025 <your-name>
# Licensed under the MIT license.

import torch
from torch import nn

from .. import baseloss
from ..reconstruction import reconstruction_loss
from .kl_div import kl_normal_loss


class Loss(baseloss.BaseLoss):
    r"""
    DIP-VAE loss (both variants) as described in

        Variational Inference of Disentangled Latent Concepts from Unlabelled
        Observations, Kumar et al., ICLR 2018.

    Parameters
    ----------
    dip_type : {"i", "ii"}, optional
        "i"  → DIP-VAE-I  (regularize Cov_{p(x)}[μ])
        "ii" → DIP-VAE-II (regularize Cov_{q(z)}[z])
    lambda_od : float, optional
        Weight for the **off-diagonal** covariance penalty λ_od.
    lambda_d  : float, optional
        Weight for the **diagonal** covariance penalty λ_d.
    beta : float, optional
        Weight of the KL term.  β = 1 gives the exact ELBO.
        (The original paper keeps β=1, but we expose it for completeness.)
    schedulers_kwargs : list[dict], optional
        Same mechanism as in betavae – lets you schedule λ’s or β if desired.

    Notes
    -----
    * Off-diagonal penalty  Σ_{i≠j} Cov²_{⋯}[·]_{ij}
    * Diagonal     penalty  Σ_i (Cov_{⋯}[·]_{ii} − 1)²
    * Everything is estimated with the current mini-batch (same trick as the paper).
    """
    def __init__(
        self,
        dip_type="i",
        lambda_od=10.0,
        lambda_d=100.0,
        beta=1.0,
        schedulers_kwargs=None,
        **kwargs,
    ):
        super().__init__(mode="post_forward",
                         schedulers_kwargs=schedulers_kwargs,
                         **kwargs)

        if dip_type not in ("i", "ii"):
            raise ValueError("dip_type must be 'i' or 'ii'.")

        # expose λ’s and β to the scheduler mechanism ------------------------
        if self.schedulers:
            if 'lambda_od' in self.schedulers:
                lambda_od = self.schedulers['lambda_od'].initial_value
            if 'lambda_d' in self.schedulers:
                lambda_d = self.schedulers['lambda_d'].initial_value
            if 'beta' in self.schedulers:
                beta = self.schedulers['beta'].initial_value

        self.dip_type   = dip_type
        self.lambda_od  = lambda_od
        self.lambda_d   = lambda_d
        self.beta       = beta

    # --------------------------------------------------------------------- #
    #                      Properties for checkpointing                     #
    # --------------------------------------------------------------------- #
    @property
    def name(self):
        return 'dipvae'

    @property
    def kwargs(self):
        base = {
            'dip_type'  : self.dip_type,
            'lambda_od' : self.lambda_od,
            'lambda_d'  : self.lambda_d,
            'beta'      : self.beta,
            'rec_dist'  : self.rec_dist,
        }
        if self.schedulers:
            base['schedulers_kwargs'] = [
                {
                    'name'      : sch.name,
                    'param_name': pname,
                    'kwargs'    : {**sch.kwargs},
                }
                for pname, sch in self.schedulers.items()
            ]
        return base

    def state_dict(self):
        if not self.schedulers:
            return None
        return {
            'scheduler_states': {
                pname: sch.state_dict() for pname, sch in self.schedulers.items()
            }
        }

    def load_state_dict(self, state_dict):
        if not state_dict or 'scheduler_states' not in state_dict:
            return
        for pname, sd in state_dict['scheduler_states'].items():
            if pname in self.schedulers:
                self.schedulers[pname].load_state_dict(sd)

    # --------------------------------------------------------------------- #
    #                              Forward                                  #
    # --------------------------------------------------------------------- #
    def __call__(self, data, reconstructions, stats_qzx, **kwargs):
        """
        Parameters
        ----------
        data : torch.Tensor              -- input batch  (B × …)
        reconstructions : torch.Tensor   -- decoded output
        stats_qzx : Tuple[μ, logσ²] or packed tensor with last dim = 2
            Statistics of qϕ(z|x) needed for KL and covariance penalties.
        """
        # ----------------------------- unpack --------------------------------
        if isinstance(stats_qzx, torch.Tensor):
            mu, logvar = stats_qzx.unbind(-1)
        else:
            mu, logvar = stats_qzx
        var = torch.exp(logvar)                    # σ²

        batch_size, latent_dim = mu.size()

        # -------------------------- base ELBO --------------------------------
        rec_loss   = reconstruction_loss(data, reconstructions,
                                         distribution=self.rec_dist)
        kl_comp    = kl_normal_loss(mu, logvar, return_components=True)
        kl_total   = kl_comp.sum()

        loss = rec_loss + self.beta * kl_total     # start with (β-)ELBO

        # ------------------ DIP-VAE covariance penalties --------------------
        # centred μ for Cov_{p(x)}[μϕ(x)]
        mu_c = mu - mu.mean(dim=0, keepdim=True)
        cov_mu = torch.matmul(mu_c.t(), mu_c) / batch_size   # d × d

        if self.dip_type == "i":
            cov_target = cov_mu                                  # Eq.(6)
        else:
            # Cov_q(z) = Cov_p[μ] + E_p[Σ]
            cov_target = cov_mu + torch.diag_embed(var.mean(dim=0))  # Eq.(7)

        off_diag_mask = torch.ones_like(cov_target, dtype=torch.bool)
        off_diag_mask.fill_diagonal_(False)

        off_diag_loss = (cov_target[off_diag_mask] ** 2).sum()
        diag_loss     = ((torch.diag(cov_target) - 1.0) ** 2).sum()

        # scale and add
        dip_reg = self.lambda_od * off_diag_loss + self.lambda_d * diag_loss
        loss    = loss + dip_reg

        # --------------------------- logging ---------------------------------
        to_log = {
            "loss"        : loss.item(),
            "rec_loss"    : rec_loss.item(),
            "kl_loss"     : kl_total.item(),
            "dip_reg"     : dip_reg.item(),
        }
        if self.log_kl_components:
            for i, c in enumerate(kl_comp):
                to_log[f"kl_loss_{i}"] = c.item()

        return {"loss": loss, "to_log": to_log}
