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
from . import hydrodynamics

import numpy as np
from numba import jit

@jit(nopython = True)
def pressure(NRT, Vir, area):

    p = ((NRT - Vir / 2) / area)
    p *= (100/NA) #Richtige Skalierung

    return p

def lennard_jones(frame, nstlist, ref_pos, conf_pos, pbc_dim, buffer_radius, vdw_cutoff, pairlist, A12, B6, offsets):

    """
    Compute the Lennard-Jones forces acting on each particle in the system.

    This function calculates inter-particle forces based on the Lennard-Jones potential, using a pairlist
    to optimize performance. The pairlist is either generated or updated depending on the current simulation frame.

    Functions Called:
    -----------------
    1. `force.generate_pairlist`:
       - Generates the pairlist, which contains pairs of particles within the cutoff distance,
         considering periodic boundary conditions (PBC).
    2. `force.update_pairlist`:
       - Updates the existing pairlist for subsequent frames, ensuring pairs within the cutoff distance are maintained.
    3. `force.calculate_force`:
       - Computes the Lennard-Jones forces for all particle pairs in the pairlist based on their distances.

    Returns:
    --------
    force_per_particle : numpy.ndarray
        A (N, 2) array representing the forces acting on each particle in the system.
    """
    
    #Generate new pair list
    if (frame % nstlist) == 1 or nstlist == 1: 
        
        #Calculate distances and distance vectors for all unique pairs of particles -> COSTLY
        dist_mat, vec_mat = utils.distance_matrix_NxN(pos = conf_pos, N = conf_pos.shape[0], pbc_dim = pbc_dim, offsets = offsets)

        #### BUFFER RADIUS
        #Pairlist       -> Contains everything that is inside the buffer radius of a particle
        pairlist_rij, pairlist_rij_sqrt, pairlist, _, _, _ = generate_pairlist(dist_mat= dist_mat, vec_mat = vec_mat, lj_buffer = buffer_radius)

    #Update distances in the self.pairlist
    else: 

        #### BUFFER RADIUS
        #Update pairlist -> Update every distance and distance vector inside the buffer radius of a particle
        pairlist_rij, pairlist_rij_sqrt = update_pairlist(ref_pos   = ref_pos, conf_pos  = conf_pos, pairlist  = pairlist, pbc_dim   = pbc_dim, offsets   = offsets[ pairlist[:, 0], pairlist[:, 1] ])
    
    #### INNER RADIUS
    #Obtain effective distances and distance vectors that are taking into account for the LJ interaction between particles
    effective_rij, effective_rij_sqrt, effective_mask, _, _ = filter_pairlist(cutoff = vdw_cutoff, rij = pairlist_rij, rij_sq = pairlist_rij_sqrt)
    

    if not effective_rij_sqrt.size > 0: return np.zeros_like( conf_pos, dtype = np.float32 ), pairlist, 0, 0
    assert effective_rij_sqrt.min() >= 1E-12, f'Too small! {effective_rij_sqrt.min()}'

    masked_pairlist = pairlist[ effective_mask ]
    
    force_per_particle, virial, pote = calculate_force(rij             = effective_rij,
                                                       rij_sq          = effective_rij_sqrt**2,
                                                       N               = ref_pos.shape[0],
                                                       lj_A12          = A12[ masked_pairlist[:, 0], masked_pairlist[:, 1] ],
                                                       lj_B6           = B6[ masked_pairlist[:, 0], masked_pairlist[:, 1] ],
                                                       masked_pairlist = masked_pairlist)
 

    return force_per_particle, pairlist, virial, pote

def lennard_jones_hydro(frame, nstlist, ref_pos, conf_pos, pbc_dim, buffer_radius, vdw_cutoff, pairlist, A12, B6, offsets, hydrodyn = False, cutoff_a = 0, cutoff_2a = 0, viscosity_scale = 0, Dij = 0):

    """
    Compute the Lennard-Jones forces acting on each particle in the system.

    This function calculates inter-particle forces based on the Lennard-Jones potential, using a pairlist
    to optimize performance. The pairlist is either generated or updated depending on the current simulation frame.

    Functions Called:
    -----------------
    1. `force.generate_pairlist`:
       - Generates the pairlist, which contains pairs of particles within the cutoff distance,
         considering periodic boundary conditions (PBC).
    2. `force.update_pairlist`:
       - Updates the existing pairlist for subsequent frames, ensuring pairs within the cutoff distance are maintained.
    3. `force.calculate_force`:
       - Computes the Lennard-Jones forces for all particle pairs in the pairlist based on their distances.

    Returns:
    --------
    force_per_particle : numpy.ndarray
        A (N, 2) array representing the forces acting on each particle in the system.
    """
    
    #Generate new pair list
    if (frame % nstlist) == 1 or nstlist == 1: 
        
        #Calculate distances and distance vectors for all unique pairs of particles -> COSTLY
        dist_mat, vec_mat = utils.distance_matrix_NxN(pos = conf_pos, N = conf_pos.shape[0], pbc_dim = pbc_dim, offsets = offsets)

        #### BUFFER RADIUS
        #Pairlist       -> Contains everything that is inside the buffer radius of a particle
        #Outer Pairlist -> Contains everything that is outside the buffer radius of a particle
        pairlist_rij, pairlist_rij_sqrt, pairlist, outer_pairlist_rij, outer_pairlist_rij_sqrt, outer_pairlist = generate_pairlist(dist_mat  = dist_mat, vec_mat   = vec_mat, lj_buffer = buffer_radius)

        #Hydrodynamics requested
        #Calculate Diffusion coefficients between particles outside the buffer radius

        #Update diffusion coefficients larger than buffer radius
        rpy_far        = hydrodynamics.rpy_far(rij_sq = outer_pairlist_rij_sqrt, rij = outer_pairlist_rij, a = cutoff_a, viscosity_scale = viscosity_scale)

        Dij[outer_pairlist[:, 0], outer_pairlist[:, 1]] = rpy_far
        Dij[outer_pairlist[:, 1], outer_pairlist[:, 0]] = rpy_far

    #Update distances in the self.pairlist
    else: 

        #Update pairlist -> Update every distance and distance vector inside the buffer radius of a particle
        pairlist_rij, pairlist_rij_sqrt = update_pairlist(ref_pos   = ref_pos,
                                                                conf_pos  = conf_pos,
                                                                pairlist  = pairlist,
                                                                pbc_dim   = pbc_dim,
                                                                offsets   = offsets[ pairlist[:, 0], pairlist[:, 1] ])
    
    #### INNER RADIUS
    #Obtain effective distances and distance vectors that are taking into account for the LJ interaction between particles
    effective_rij, effective_rij_sqrt, effective_mask, _, _ = filter_pairlist(cutoff = vdw_cutoff, rij = pairlist_rij, rij_sq = pairlist_rij_sqrt)
    

    #Hydrodynamics requested
    near_rij, near_rij_sqrt, near_mask, far_rij, far_rij_sqrt = filter_pairlist(cutoff = cutoff_2a, rij = pairlist_rij, rij_sq = pairlist_rij_sqrt)

    #NEAR FIELD DIFFUSION
    rpy_near        = hydrodynamics.rpy_near(rij_sq = near_rij_sqrt, rij = near_rij, a = cutoff_a, viscosity_scale = viscosity_scale)
    
    Dij[ pairlist[ near_mask, 0], pairlist[ near_mask, 1]] = rpy_near
    Dij[ pairlist[ near_mask, 1], pairlist[ near_mask, 0]] = rpy_near
    
    #FAR FIELD DIFFUSION
    rpy_far        = hydrodynamics.rpy_far(rij_sq = far_rij_sqrt, rij = far_rij, a = cutoff_a, viscosity_scale = viscosity_scale)

    Dij[ pairlist[ ~near_mask, 0], pairlist[ ~near_mask, 1]] = rpy_far
    Dij[ pairlist[ ~near_mask, 1], pairlist[ ~near_mask, 0]] = rpy_far

    #-----------------------------------------------------------------------------
    
    if not effective_rij_sqrt.size > 0: return np.zeros_like( conf_pos, dtype = np.float32 ), pairlist, Dij
    assert effective_rij_sqrt.min() >= 1E-12, f'Too small! {effective_rij_sqrt.min()}'

    force_per_particle, virial, pote = calculate_hydro_force(rij             = effective_rij,
                                                     rij_sq          = effective_rij_sqrt**2,
                                                     N               = ref_pos.shape[0],
                                                     lj_A12          = A12,
                                                     lj_B6           = B6,
                                                     masked_pairlist = pairlist[ effective_mask ], 
                                                     Dij             = Dij)

    return force_per_particle, pairlist, Dij, virial, pote

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

    sr6        = sr6  *  lj_B6
    sr12       = sr12 * lj_A12

    #Calculate potential energy
    pote       = (sr12 / 2 - sr6 ) / 6

    #Calculate the virial
    virial     = (sr12 - sr6 )

    #Calculate "scaling factor" for the force
    force      = virial * inv_rij_sq

    #Multiplicate "scaling factor" with the force direction -> This is now the force acting from particle j on particle i.
    force      = force.reshape(-1, 1) * rij

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

    return force_per_particle, virial, pote

@jit(nopython=True)
def calculate_hydro_force(rij, rij_sq, N, lj_A12, lj_B6, masked_pairlist, Dij):

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

    sr6       *= lj_B6
    sr12      *= lj_A12

    #Calculate potential energy
    pote       = (sr12 / 2 - sr6 ) / 6

    #Calculate the virial
    virial     = (sr12 - sr6 )

    #Calculate "scaling factor" for the force
    force      = virial * inv_rij_sq

    #Multiplicate "scaling factor" with the force direction -> This is now the force acting from particle j on particle i.
    force      = force.reshape(-1, 1) * rij
    force      = force.astype(np.float32)

    #Storage factor for the force per particle
    force_per_particle = np.zeros( (N, 2), dtype = np.float32 )

    #Iterate over all particle pairs in the pairlist with a pair distance below the VdW cutoff
    k = 0
    for pair in masked_pairlist:

        #Extract pair
        i, j = pair[0], pair[1]

        force_k = Dij[i,j] @ force[k] # np.array([ np.sum(Dij[i,j][0] * force[k]), np.sum(Dij[i,j][1] * force[k]) ], dtype = np.float32 )

        #Apply Newton's third law: Actio est reactio 
        force_per_particle[i] += force_k
        force_per_particle[j] -= force_k #Equal force is acting on j, therefore subtraction

        k += 1

    return force_per_particle, virial, pote

def generate_pairlist(dist_mat, vec_mat, lj_buffer):

    """
    Pair list generation. A pair list keeps track of neighboured particles to speed up the force calculation.

    Parameters
    ----------

    pos       := numpy.ndarray
        Position of the particles. Expected are 2-dimensional positions.
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

    #Get indices of pair distances below outer cutoff radius
    pairlist       = np.where(  dist_mat <= lj_buffer )
    outer_pairlist = np.where( (dist_mat > lj_buffer) & (dist_mat < np.inf) )

    #Convert it to two-dimensional array -> (K, 2) with K the number of particle pairs with distance below lj_buffer
    pairlist       = np.vstack(       pairlist ).T
    outer_pairlist = np.vstack( outer_pairlist ).T

    #Extract distance vector and squared distances based on the pair list
    rij    =  vec_mat[ pairlist[:, 0], pairlist[:, 1] ]
    rij_sq = dist_mat[ pairlist[:, 0], pairlist[:, 1] ]
    
    outer_rij    =  vec_mat[ outer_pairlist[:, 0], outer_pairlist[:, 1] ]
    outer_rij_sq = dist_mat[ outer_pairlist[:, 0], outer_pairlist[:, 1] ]

    return rij, rij_sq, pairlist, outer_rij, outer_rij_sq, outer_pairlist
    
@jit(nopython=True)
def filter_pairlist(cutoff, rij, rij_sq):

    mask = (rij_sq <= cutoff)

    inner_rij   , outer_rij    = rij[mask],    rij[~mask]
    inner_rij_sq, outer_rij_sq = rij_sq[mask], rij_sq[~mask]

    return inner_rij, inner_rij_sq, mask, outer_rij, outer_rij_sq

@jit(nopython=True)
def update_pairlist(ref_pos, conf_pos, pairlist, pbc_dim, offsets):

    """
    This function is called if a pairlist was generated in a previous step.

    Parameters
    ----------
    ref_pos       := numpy.ndarray
        Position of the reference particles. Expected are 2-dimensional positions.
    conf_pos       := numpy.ndarray
        Position of the configuration particles. Expected are 2-dimensional positions.
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

    # Get the maximum valid index
    max_index_N = ref_pos.shape[0] - 1
    max_index_M = conf_pos.shape[0] - 1
    
    # Check each index in the pairlist
    for idx in pairlist[:, 0]:
        if idx < 0 or idx > max_index_N:
            # Raise an IndexError with a descriptive message
            raise IndexError(f"Index {idx} is out of bounds for arrays with {ref_pos.shape} elements")
    
    # Check each index in the pairlist
    for idx in pairlist[:, 1]:
        if idx < 0 or idx > max_index_M:
            # Raise an IndexError with a descriptive message
            raise IndexError(f"Index {idx} is out of bounds for arrays with {conf_pos.shape} elements")
    
    
    #Current distances between particle pairs in pairlist
    #Distance vector points from second column to first column -> j -> i, because first column denotes row of the NxN distance matrix
    rij    = ref_pos[ pairlist[:, 0] ] - conf_pos[ pairlist[:, 1]]
    
    #Apply periodic boundary conditions
    for k, size in zip(pbc_dim[0], pbc_dim[1]):
        rij[:, k] = np.where(rij[:, k] >    size / 2, rij[:, k] - size, rij[:, k])
        rij[:, k] = np.where(rij[:, k] <= - size / 2, rij[:, k] + size, rij[:, k])

    #Calculate squared distances
    
    rij_       = np.sqrt( np.sum(rij**2,axis=1) )
    rij_offset = rij_ - offsets

    #Store the squared distances and distance vectors in the arrays
    return ( rij_offset / rij_ ).reshape(-1, 1) * rij, rij_offset
