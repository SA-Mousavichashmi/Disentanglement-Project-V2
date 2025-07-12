"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# Only include implemented and valid loss names
LOSS_LIST = ['betavae', 'annealedvae', 'factorvae', 'betatcvae', 'beta_toroidal_vae', 'group_theory', 'beta_s_n_vae', 'dipvae']
RECON_DISTS = ["bernoulli", "laplace", "gaussian"]

def select(name, **kwargs):
    """Return the correct loss function given the arguments."""
    if name == "betavae":
        from losses.n_vae.betavae import Loss
        return Loss(**kwargs)
    if name == "annealedvae":
        from losses.n_vae.annealedvae import Loss
        return Loss(**kwargs)
    if name == "factorvae":
        from losses.n_vae.factorvae import Loss
        return Loss(**kwargs)
    if name == "betatcvae":
        from losses.n_vae.betatcvae import Loss
        return Loss(**kwargs)
    if name == 'beta_toroidal_vae':
        from losses.s_vae.beta_toroidal_vae import BetaToroidalVAELoss
        return BetaToroidalVAELoss(**kwargs)
    if name == 'group_theory':
        from losses.group_theory.group_theory import Loss
        return Loss(**kwargs)
    if name == 'beta_s_n_vae':
        from losses.s_n_vae.beta_s_n_vae import BetaSNVAELoss
        return BetaSNVAELoss(**kwargs)
    if name == 'dipvae':
        from losses.n_vae.dipvae import Loss
        return Loss(**kwargs)
    
    err = "Unknown loss.name = {}. Possible values: {}"
    raise ValueError(err.format(name, LOSS_LIST))
