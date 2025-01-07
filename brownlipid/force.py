# ----PYTHON---- #
"""
force.py

This module implements functions for calculating external forces using 
the Lennard-Jones potential.

Functions include:
- Computing the Lennard-Jones potential.
- Calculating force magnitudes and directions.
- Utilities for handling pairwise interactions.

Author: Marius Trollmann
"""

from . import utils

import numpy as np
from numba import jit

@jit(nopython=True)
def calculate_force(rij, rij_sq, N, lj_A12, lj_B6, masked_pairlist):
    
    inv_rij_sq = 1.0 / rij_sq

    #Calculate powers
    sr6  = inv_rij_sq ** 3
    sr12 = sr6 ** 2

    force = (lj_A12 * sr12 - lj_B6 * sr6 ) * inv_rij_sq 
    force = force.reshape(-1, 1) * rij

    force_per_particle = np.zeros( (N, 2), dtype = np.float32 )

    k = 0
    for pair in masked_pairlist:

        i, j = pair[0], pair[1]

        force_per_particle[i] += force[k]
        force_per_particle[j] -= force[k]

        k += 1

    return force_per_particle

def generate_pairlist(pos, N, pbc_dim, lj_buffer, lj_cutoff):

    #------------------------------------------------
    #Setup pair list

    dist_mat, vec_mat = utils.distance_matrix_NxN(pos = pos, N = N, pbc_dim = pbc_dim)

    pairlist = np.vstack( np.where( dist_mat <= lj_buffer ) ).T

    #First column always smaller than second column
    pairlist = np.sort(pairlist, axis = 1)

    pairlist = np.unique(pairlist, axis = 0)
    
    #------------------------------------------------
    #Buffer

    rij    =  vec_mat[ pairlist[:, 0], pairlist[:, 1] ]
    rij_sq = dist_mat[ pairlist[:, 0], pairlist[:, 1] ]

    assert np.all(np.isfinite(rij))   , 'Inf or nan in vector matrix!'
    assert np.all(np.isfinite(rij_sq)), 'Inf or nan in distance matrix!'

    #------------------------------------------------
    #Cutoff
    mask = (rij_sq <= lj_cutoff)

    rij    = rij[mask]
    rij_sq = rij_sq[mask]

    return rij, rij_sq, mask, pairlist

@jit(nopython=True)
def update_pairlist(pos, pairlist, pbc_dim, lj_cutoff):
    
    rij    = pos[ pairlist[:, 0] ] - pos[ pairlist[:, 1]]
    
    #Apply periodic boundary conditions
    for j, size in zip(pbc_dim[0], pbc_dim[1]):
        rij[:, j] = np.where(rij[:, j] >    size / 2, rij[:, j] - size, rij[:, j])
        rij[:, j] = np.where(rij[:, j] <= - size / 2, rij[:, j] + size, rij[:, j])

    
    rij_sq = np.sum(rij**2, axis = 1)
    
    mask = (rij_sq <= lj_cutoff)

    vec_r      = rij[mask]
    vec_r_norm = rij_sq[mask]

    return vec_r, vec_r_norm, mask
