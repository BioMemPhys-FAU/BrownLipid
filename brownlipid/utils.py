# ----PYTHON---- #
"""
utils.py

This module contains a bunch of helper functions that are used in multiple steps of the main workflow.

"""


import numpy as np
import matplotlib.pyplot as plt

from numba import jit

@staticmethod
def heaviside(x, threshold):

    """
    Custom Heaviside Function.
    Returns 1, if x is smaller or equal than the threshold.
    Returns 0, if x is greater than the threshold.
    
    Parameters
    ----------

    x := float
        Float that is evaluated
    threshold := float
        Float that is used a constant for the evaluation

    Returns
    -------

    int := 0 or 1
    """

    if x <= threshold: return 1
    else: return 0

@jit(nopython=True)
def distance_matrix_NxN(pos, N, pbc_dim):

    """
    Calculate distance matrix between all possible pairs of particles.
    This function is computational expensive, and for the sake of performance it should be called as rarley as possible.
    However, increasing the call frequency could improve the accuracy of the simulation.
    
    Improvements taken from:
    https://github.com/Allen-Tildesley/examples/blob/master/python_examples/md_lj_module.py

    Parameters
    ----------

    pos     := numpy.ndarray
        Positional vector. Expects two-dimensional coordinates -> (N, 2)
    N       := int
        Number of particles in the system.
    pbc_dim := tuple
        First list contains index of dimensions along which PBC is applied. Second list contains length of box vector along which PBC is applied.

    Returns
    -------

    vec_mat  := numpy.ndarray
        Squared distances between particles.
    dist_mat := numpy.ndarray
        Distance vectors between particles.


    """

    #Init storage vectors
    dist_mat = np.ones((N, N)   , dtype = np.float32) * np.inf
    vec_mat  = np.ones((N, N, 2), dtype = np.float32) * np.inf

    #Fill only upper triangle
    for i in range(N - 1):

        #rij is the vector that points from (i+1) to i, or otherwise that points from j to i.
        rij    = pos[i, :] - pos[(i+1):, :] 

        #Apply periodic boundary conditions
        for k, size in zip(pbc_dim[0], pbc_dim[1]):
            rij[:, k] = np.where(rij[:, k] >    size / 2, rij[:, k] - size, rij[:, k])
            rij[:, k] = np.where(rij[:, k] <= - size / 2, rij[:, k] + size, rij[:, k])


        #For the force calculation later only the squared distance is required, therefore the square root operation is omitted.
        rij_sq = np.sum(rij**2,axis=1)

        #Store the squared distances and distance vectors in the arrays
        dist_mat[i, (i+1):]    = rij_sq
        vec_mat[ i, (i+1):, :] = rij

    return dist_mat, vec_mat

@jit(nopython=True)
def apply_pbc_vector(vec, pbc_dim) -> np.ndarray:

        """
        Periodic boundary conditions for DIRECTIONAL VECTORS a.k.a. VECTORS.

        If a component of a vector is larger than half the box size (positive and negative), subtract one box size.

        """
        
        assert vec.ndim == 2, 'Vector has not a dimension of 2'
        
        #Apply PBC
        for i, size in zip(pbc_dim[0], pbc_dim[1]):
            vec[:, i] = np.where(vec[:, i] >    size / 2, vec[:, i] - size, vec[:, i])
            vec[:, i] = np.where(vec[:, i] <= - size / 2, vec[:, i] + size, vec[:, i])

        return vec

@jit(nopython=True)
def check_circ_cond(pos, mid, r, pbc_dim):

    """
    Check if a particle is within a circle.

    Parameters
    ----------
    

    """

    assert mid.ndim == 2, 'Middle point is not two-dimensional'

    circ_coor = pos - mid
    #Apply periodic boundary conditions
    for j, size in zip(pbc_dim[0], pbc_dim[1]):
        circ_coor[:, j] = np.where(circ_coor[:, j] >    size / 2, circ_coor[:, j] - size, circ_coor[:, j])
        circ_coor[:, j] = np.where(circ_coor[:, j] <= - size / 2, circ_coor[:, j] + size, circ_coor[:, j])


    circ_dist = np.sum(circ_coor**2, axis = 1)

    return np.where(circ_dist <= r)[0]

@jit(nopython=True)
def check_not_circ_cond(pos, mid, r, pbc_dim):

    """
    Check if a particle is within a circle.

    Parameters
    ----------
    

    """

    assert mid.ndim == 2, 'Middle point is not two-dimensional'

    circ_coor = pos - mid
    #Apply periodic boundary conditions
    for j, size in zip(pbc_dim[0], pbc_dim[1]):
        circ_coor[:, j] = np.where(circ_coor[:, j] >    size / 2, circ_coor[:, j] - size, circ_coor[:, j])
        circ_coor[:, j] = np.where(circ_coor[:, j] <= - size / 2, circ_coor[:, j] + size, circ_coor[:, j])


    circ_dist = np.sum(circ_coor**2, axis = 1)

    return np.where(circ_dist > r)[0]
    

@jit(nopython=True)
def check_square_cond(self, pos, Lx, Ly, mid, pbc_dim): 
    
    assert mid.ndim == 2, 'Middle point is not two-dimensional'

    squa_coor = (pos - mid)
    #Apply periodic boundary conditions
    for j, size in zip(pbc_dim[0], pbc_dim[1]):
        squa_coor[:, j] = np.where(squa_coor[:, j] >    size / 2, squa_coor[:, j] - size, squa_coor[:, j])
        squa_coor[:, j] = np.where(squa_coor[:, j] <= - size / 2, squa_coor[:, j] + size, squa_coor[:, j])

    cond_x = np.logical_and( (-Lx/2 <= squa_coor[:, 0]), (squa_coor[:, 0] <= Lx/2) )
    cond_y = np.logical_and( (-Ly/2 <= squa_coor[:, 1]), (squa_coor[:, 1] <= Ly/2) )

    return np.where( np.logical_and(cond_x, cond_y))[0]


@jit(nopython=True)
def perpDot(a, b):
    
    a_vert = np.copy(a)
    a_vert = np.flip(a_vert, axis = 1)
    a_vert[:, 0] = -1 * a_vert[:, 0]

    return np.dot(a_vert, b)

def test_intersection(intersection, curr_points, prev_points, displace, pbc_dim):

    #-------------------------------------
    #Check if intersection is between old positions and new positions
    #The following code checks if the intersection is located on the shortest vector between the previous and the
    #current position based on the smallest image convention.

    intersection = intersection.reshape(-1, 2)
    curr_points  = curr_points.reshape(-1, 2)
    prev_points  = prev_points.reshape(-1, 2)
    displace     = displace.reshape(-1, 2)

    #Vector from current positions to intersection
    curr_inter  = intersection - curr_points
    curr_inter  = apply_pbc_vector(vec = curr_inter, pbc_dim = pbc_dim)

    #Vector from previous positions to intersection
    prev_inter  = intersection - prev_points
    prev_inter  = apply_pbc_vector(vec = prev_inter, pbc_dim = pbc_dim)

    #Using squares and Pythagoras Theorem
    curr_inter2 = np.sum((curr_inter)**2, axis = 1)
    prev_inter2 = np.sum((prev_inter)**2, axis = 1)
    d2          = np.sum(    displace**2, axis = 1)

    check = curr_inter2 + prev_inter2 + 2 * np.sqrt(curr_inter2 * prev_inter2)

    return check, d2
