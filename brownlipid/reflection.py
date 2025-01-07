import numpy as np
from numba import jit

@jit(nopython=True)
def get_normals_circle(prev_points, points, d, mid, r, pbc_dim):

    """
    Calculate normal vector of a two dimensional circle for a point, and the intersection with the circular boundary.

    The required normal vector here is the vector from the circle midpoint to the intersection point, where the particle gets reflected.

    The intersection is calculated solving the linear equation:

        || M + d * lambda || = r

    , where M is the directional vector between the circle midpoint and the previous position; d is the displacement vector between the previous
    and the current position; lambda is a scale factor for the displacement vector; and r is the radius of the circle.
    The linear equation has two solutions. Here, always the smaller value of lambda is assumed to be the required value.

    Parameters
    ----------
    points := numpy.ndarray
        coordinates of reflected points
    prev_points := numpy.ndarray
        previous coordinates of reflected points
    d := numpy.ndarray
        displace vector
    mid := numpy.ndarray
        circle midpoint
    r := float
        radius of the circle
    
    """

    #Previous and current positions -> both are wrapped!
    prev_points = prev_points.reshape(-1,2)
    points      = points.reshape(-1,2)

    #-------------------------------------
    #Solve linear equation
    M = (prev_points - mid)
    #Apply periodic boundary conditions
    for j, size in zip(pbc_dim[0], pbc_dim[1]):
        M[:, j] = np.where(M[:, j] >    size / 2, M[:, j] - size, M[:, j])
        M[:, j] = np.where(M[:, j] <= - size / 2, M[:, j] + size, M[:, j])
    
    A = d[:, 0]**2 + d[:, 1]**2
    B = M[:, 0] * d[:, 0] + M[:, 1] * d[:, 1]
    C = M[:, 0]**2 + M[:, 1]**2

    lam1 = 2 * B + 2 * np.sqrt(B**2 - A * ( C - r**2 ) )
    lam1 /= 2 * A
    
    lam2 = 2 * B - 2 * np.sqrt(B**2 - A * ( C - r**2 ) )
    lam2 /= 2 * A

    lam1 = np.abs(lam1)
    lam2 = np.abs(lam2)

    lam = np.zeros( lam1.shape )

    lam[lam1 < lam2] = lam1[lam1 < lam2]
    lam[lam2 < lam1] = lam2[lam2 < lam1]

    lam = lam.reshape(-1, 1)

    assert lam.shape[0] == points.shape[0], 'Lambda has the wrong shape'
    
    #Calculate intersection
    intersection = prev_points + lam * d
    #Apply periodic boundary conditions
    for j, size in zip(pbc_dim[0], pbc_dim[1]):
        intersection[:, j] = np.where(intersection[:, j] >    size / 2, intersection[:, j] - size, intersection[:, j])
        intersection[:, j] = np.where(intersection[:, j] <= - size / 2, intersection[:, j] + size, intersection[:, j])

    #-------------------------------------
    #Calculate normal vector
    norm = intersection - mid

    #Apply periodic boundary conditions
    for j, size in zip(pbc_dim[0], pbc_dim[1]):
        norm[:, j] = np.where(norm[:, j] >    size / 2, norm[:, j] - size, norm[:, j])
        norm[:, j] = np.where(norm[:, j] <= - size / 2, norm[:, j] + size, norm[:, j])
    
    #Normalize
    norm /= np.sqrt( np.sum(norm**2, axis = 1) ).reshape(-1, 1)

    return norm, intersection

@jit(nopython=True)
def calc_reflection(intersection, p_in, n, pbc_dim):

    """
    Calculate reflection vector of vector d with normal n

    d := numpy.ndarray
        displacement vectors of particles that are reflected
    n := numpy.ndarray
        normal of geometry
    """

    #Vectorized form
    #d.shape = (Nr, 2)
    #n.shape = (n , 2)
    
    d_rest = p_in - intersection
    
    #Apply periodic boundary conditions
    for j, size in zip(pbc_dim[0], pbc_dim[1]):
        d_rest[:, j] = np.where(d_rest[:, j] >    size / 2, d_rest[:, j] - size, d_rest[:, j])
        d_rest[:, j] = np.where(d_rest[:, j] <= - size / 2, d_rest[:, j] + size, d_rest[:, j])

    reflection_vector =  d_rest - 2 * np.sum(d_rest * n, axis = 1).reshape(-1, 1) * n
    
    reflection = intersection + reflection_vector

    #Apply periodic boundary conditions
    for j, size in zip(pbc_dim[0], pbc_dim[1]):
        reflection[:, j] = np.where(reflection[:, j] >    size / 2, reflection[:, j] - size, reflection[:, j])
        reflection[:, j] = np.where(reflection[:, j] <= - size / 2, reflection[:, j] + size, reflection[:, j])

    """
    #----------------------------------------------------
    #Test reflection
    pos_p_prev = -1 * d_rest
    pos_ref    = utils.apply_pbc_vector(reflection_vector)

    dot_pp = np.sum(pos_p_prev  * n, axis = 1)
    dot_pr = np.sum(pos_ref     * n, axis = 1)

    dot_pp /= np.linalg.norm(pos_p_prev, axis = 1)
    dot_pr /= np.linalg.norm(pos_ref   , axis = 1)

    if not np.allclose(dot_pr, dot_pp) and not np.allclose(dot_pp, dot_pr):

        print(dot_pr)
        print(dot_pp)
        
        print('Intersection:')
        print(intersection)

        raise ValueError("In angle is not equal out angle")
    #----------------------------------------------------
    """

    return reflection, reflection_vector
