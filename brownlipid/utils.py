# ----PYTHON---- #
#Helper functions


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

    #Improvements taken from:
    #https://github.com/Allen-Tildesley/examples/blob/master/python_examples/md_lj_module.py

    dist_mat = np.ones((N, N)   , dtype = np.float32) * np.inf
    vec_mat  = np.ones((N, N, 2), dtype = np.float32) * np.inf

    #Fill only upper triangle
    for i in range(N - 1):

        rij    = pos[i, :] - pos[(i+1):, :] 

        #Apply periodic boundary conditions
        for j, size in zip(pbc_dim[0], pbc_dim[1]):
            rij[:, j] = np.where(rij[:, j] >    size / 2, rij[:, j] - size, rij[:, j])
            rij[:, j] = np.where(rij[:, j] <= - size / 2, rij[:, j] + size, rij[:, j])


        rij_sq = np.sum(rij**2,axis=1)

        dist_mat[i, (i+1):]    = rij_sq
        vec_mat[ i, (i+1):, :] = rij

    return dist_mat, vec_mat

@jit(nopython=True)
def apply_pbc_vector(vec, pbc_dim) -> np.ndarray:

        """
        Periodic boundary conditions for DIRECTIONAL VECTORS a.k.a. VECTORS.

        If a component of a vector is larger than half the box size (positive and negative), subtract one box size.

        """
        
        if not any( pbc_dim ): return vec

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


    circ_dist = np.sqrt( np.sum(circ_coor**2, axis = 1) )

    return np.where(circ_dist < r)[0]
    

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
