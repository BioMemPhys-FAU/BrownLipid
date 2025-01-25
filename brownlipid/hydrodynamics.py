import numpy as np
from numba import jit

@jit(nopython=True)
def get_R(L, N, dt):
#def get_R(Dij, N, dt):

    #L = np.linalg.cholesky(Dij)

    R = np.zeros( (N, 2), dtype = np.float32)

    for i in range(N):

        for j in range(i):

            g =np.random.randn(2).astype(np.float32)

            R[i][0] = np.sum(L[i, j][0] * g)
            R[i][1] = np.sum(L[i, j][1] * g)

    R = R * np.sqrt(2 * dt )

    return R

def dyadic_product(rij):

    return np.einsum('ij,ik->ijk', rij, rij)

@jit(nopython=True)
def rpy_far(rij_rij, r, a):

    I = np.eye(2)

    far_field  = ( I + rij_rij ) + 2 * (a/ r)**2 * ( I / 3 - rij_rij )

    far_field  /= (8 * r )

    return far_field


@jit(nopython=True)
def rpy_near(rij_rij, r, a):
    
    I = np.eye(2)

    ra = 3 * r / a / 32
    
    near_field = ( 1 - 3 * ra ) * I + ra * rij_rij
    
    near_field /= (6 * a)

    return near_field




