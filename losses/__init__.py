"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

LOSS_LIST = ['betavae', 'annealedvae', 'vae', 'factorvae', 'betatcvae', 'adagvae', 'factorizedsupportvae', 'factorizedsupporttcvae']
RECON_DISTS = ["bernoulli", "laplace", "gaussian"]

def select(name, **kwargs):
    """Return the correct loss function given the arguments."""
    if name == "betavae":
        from losses.betavae import Loss
        return Loss(**kwargs)
    if name == "vae":
        from losses.betavae import Loss
        return Loss(beta=1, **kwargs)
    if name == "annealedvae":
        from losses.annealedvae import Loss
        return Loss(**kwargs)
    if name == "factorvae":
        from losses.factorvae import Loss
        return Loss(**kwargs)
    if name == "betatcvae":
        from losses.betatcvae import Loss
        return Loss(**kwargs)
    if name == "adagvae":
        from losses.adagvae import Loss
        return Loss(**kwargs)
    if name == 'factorizedsupportvae':
        from losses.factorizedsupportvae import Loss
        return Loss(**kwargs)
    if name == 'factorizedsupporttcvae':
        from losses.factorizedsupporttcvae import Loss
        return Loss(**kwargs)
    err = "Unknown loss.name = {}. Possible values: {}"
    raise ValueError(err.format(name, LOSS_LIST))
