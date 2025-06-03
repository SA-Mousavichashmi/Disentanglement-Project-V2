# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
S-N-VAE models that combine spherical (S1) and normal (R1) latent factor topologies.
"""

from .s_n_vae_locatello import Model as SNVAELocatello

__all__ = ['SNVAELocatello']
