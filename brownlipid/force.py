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

    """
    Core function for force calculation.

    The function calculate the pair-wise additive forces between particles derived from a Lennard-Jones potential

        V(r) = 4 * eps * ( (sig/r)^12 - (sig/r)^6 )         (1)

    r is the distance between two particles; sig and eps are parameters of the Lennard-Jones potential pre-defined by the user.

    Parameters
    ----------

    rij             := numpy.ndarray
        Directional vectors from particle j to i.
    rij_sq          := numpy.ndarray
        Squared distances between particle j and i.
    N               := int
        Number of particles in the system.
    lj_A12          := float
        Lennard Jones parameter for the repulsive part. User-defined.
    lj_B6           := float
        Lennard Jones parameter for the attractive part. User-defined.
    masked_pairlist := numpy.ndarray
        Sub-section of a larger pairlist. Contains only pairs with a distance below the VdW cutoff.


    """
    
    #Calculate inverse of the squared distance
    inv_rij_sq = 1.0 / rij_sq

    #Calculate powers for the attractive and the repulsive part of the Lennard-Jones potential
    sr6        = inv_rij_sq ** 3
    sr12       = sr6 ** 2

    #Calculate "scaling factor" for the force
    force = (lj_A12 * sr12 - lj_B6 * sr6 ) * inv_rij_sq

    #Multiplicate "scaling factor" with the force direction -> This is now the force acting from particle j on particle i.
    force = force.reshape(-1, 1) * rij

    #Storage factor for the force per particle
    force_per_particle = np.zeros( (N, 2), dtype = np.float32 )

    #Iterate over all particle pairs in the pairlist with a pair distance below the VdW cutoff
    k = 0
    for pair in masked_pairlist:

        #Extract pair
        i, j = pair[0], pair[1]

        #Apply Newton's third law: Actio est reactio 
        force_per_particle[i] += force[k] #Force is acting on i, therefore addition
        force_per_particle[j] -= force[k] #Equal force is acting on j, therefore subtraction

        k += 1

    return force_per_particle

def generate_pairlist(pos, N, pbc_dim, lj_buffer, lj_cutoff):

    """
    Pair list generation. A pair list keeps track of neighboured particles to speed up the force calculation.

    Parameters
    ----------

    pos       := numpy.ndarray
        Position of the particles. Expected are 2-dimensional positions.
    N         := int
        Number of particles in the system.
    pbc_dim   := tuple
        Contains two list. The first list contains indices of axes with PBC. The second list contains the length of the corresponding axes.
    lj_buffer := float
        Outer cutoff. Particle pairs with a distance below lj_buffer are stored in the pairlist. User-defined.
    lj_cutoff := float
        Inner cutoff or VdW cutoff. Particle pairs with a distance below lj_cufoff are used for force calculation. User-defined.

    Returns
    -------
    rij      := numpy.ndarray
        Distance vectors of unique particles pairs with a distance below lj_cutoff.
    rij_sq   := numpy.ndarray
        Distances between unique particles pairs with a distance below lj_cutoff.
    mask     := numpy.ndarray
        Boolean mask for pairlist. Maps to unique particle pairs with a distance below lj_cutoff.
    pairlist := numpy.ndarray
        Array containing unique particles pairs with a distance below lj_buffer.

    """

    #------------------------------------------------
    #Setup pair list

    #Calculate distances and distance vectors for all unique pairs of particles 
    dist_mat, vec_mat = utils.distance_matrix_NxN(pos = pos, N = N, pbc_dim = pbc_dim)

    #Get indices of pair distances below outer cutoff radius
    pairlist = np.where( dist_mat <= lj_buffer )
    
    #Convert it to two-dimensional array -> (K, 2) with K the number of particle pairs with distance below lj_buffer
    pairlist = np.vstack( pairlist ).T

    #I don't think I need the two lines, because dist_mat includes only the upper triangle of the full distance matrix
    #First column always smaller than second column
    #pairlist = np.sort(pairlist, axis = 1)
    #pairlist = np.unique(pairlist, axis = 0)
    
    #------------------------------------------------
    #Buffer

    #Extract distance vector and squared distances based on the pair list
    rij    =  vec_mat[ pairlist[:, 0], pairlist[:, 1] ]
    rij_sq = dist_mat[ pairlist[:, 0], pairlist[:, 1] ]

    assert np.all(np.isfinite(rij))   , 'Inf or nan in vector matrix!'
    assert np.all(np.isfinite(rij_sq)), 'Inf or nan in distance matrix!'

    #------------------------------------------------
    #Apply inner cutoff
    mask = (rij_sq <= lj_cutoff)

    rij    = rij[mask]
    rij_sq = rij_sq[mask]

    return rij, rij_sq, mask, pairlist

@jit(nopython=True)
def update_pairlist(pos, pairlist, pbc_dim, lj_cutoff):

    """
    This function is called if a pairlist was generated in a previous step.

    Parameters
    ----------
    pos       := numpy.ndarray
        Position of the particles. Expected are 2-dimensional positions.
    pairlist  := numpy.ndarray
        Full pairlist. Contains unique particle pairs that had a distance below lj_buffer during its generation.
    pbc_dim   := tuple
        Contains two list. The first list contains indices of axes with PBC. The second list contains the length of the corresponding axes.
    lj_cutoff := float
        Inner cutoff or VdW cutoff. Particle pairs with a distance below lj_cufoff are used for force calculation. User-defined.

    Returns
    -------    
    rij      := numpy.ndarray
        Distance vectors of unique particles pairs with a distance below lj_cutoff.
    rij_sq   := numpy.ndarray
        Distances between unique particles pairs with a distance below lj_cutoff.
    mask     := numpy.ndarray
        Boolean mask for pairlist. Maps to unique particle pairs with a distance below lj_cutoff.

    """
    
    #Current distances between particle pairs in pairlist
    #Distance vector points from second column to first column -> j -> i, because first column denotes row of the NxN distance matrix
    rij    = pos[ pairlist[:, 0] ] - pos[ pairlist[:, 1]]
    
    #Apply periodic boundary conditions
    for k, size in zip(pbc_dim[0], pbc_dim[1]):
        rij[:, k] = np.where(rij[:, k] >    size / 2, rij[:, k] - size, rij[:, k])
        rij[:, k] = np.where(rij[:, k] <= - size / 2, rij[:, k] + size, rij[:, k])

    #Calculate squared distances
    rij_sq = np.sum(rij**2, axis = 1)
    
    #Create a boolean mask for particle pairs with distances below VdW cutoff
    mask = (rij_sq <= lj_cutoff)
    
    #Apply boolean mask
    rij    = rij[mask]
    rij_sq = rij_sq[mask]

    return rij, rij_sq, mask
