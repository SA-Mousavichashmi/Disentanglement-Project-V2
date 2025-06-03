"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

MODEL_LIST = [
    'vae', 
    'vae_burgess', 
    'vae_chen_mlp', 
    'vae_locatello', 
    'vae_locatello_sbd',
    'vae_montero_small', 
    'vae_montero_large',
    # S-VAE models
    'toroidal_vae',
    'toroidal_vae_burgess',
    'toroidal_vae_locatello',
    # S-N-VAE models
    's_n_vae_locatello',
]

def select(name, **kwargs):
    if name not in MODEL_LIST:
        err = "Unknown model.name = {}. Possible values: {}"
        raise ValueError(err.format(name, MODEL_LIST))

    # Assuming models are now in n_vae
    if name == 'vae_burgess':
        from .n_vae.vae_burgess import Model
        return Model(**kwargs)
    if name == 'vae_chen_mlp':
        from .n_vae.vae_chen_mlp import Model
        return Model(**kwargs)
    if name == 'vae_locatello':
        from .n_vae.vae_locatello import Model
        return Model(**kwargs)
    if name == 'vae_locatello_sbd':
        from .n_vae.vae_locatello_sbd import Model
        return Model(**kwargs)
    if name == 'vae_montero_small':
        from .n_vae.vae_montero_small import Model
        return Model(**kwargs)
    if name == 'vae_montero_large':
        from .n_vae.vae_montero_large import Model
        return Model(**kwargs)
    if name == 'vae': # Assuming 'vae' refers to vae_locatello in n_vae
        from .n_vae.vae_locatello import Model
        return Model(**kwargs)
    # Add logic for s_vae models here
    if name == 'toroidal_vae_burgess':
        from .s_vae.toroidal_vae.toroidal_vae_burgess import Model
        return Model(**kwargs)
    if name == 'toroidal_vae_locatello':
        from .s_vae.toroidal_vae.toroidal_vae_locatello import Model
        return Model(**kwargs)
    if name == 'toroidal_vae': # Generic toroidal VAE with selectable encoder/decoder
        from .s_vae.toroidal_vae.toroidal_vae import Model
        return Model(**kwargs)
    # Add logic for s_n_vae models here
    if name == 's_n_vae_locatello':
        from .s_n_vae.s_n_vae_locatello import Model
        return Model(**kwargs)
