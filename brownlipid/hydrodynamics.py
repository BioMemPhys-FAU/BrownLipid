import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel = True)
def numba_cholesky(a):

    L = np.zeros_like(a)
    for i in prange(a.shape[0]):
        for j in range(i, a.shape[1]):
            inter = np.linalg.cholesky(a[i,j])
            L[i,j], L[j,i] = inter, inter
            
    return L

@jit(nopython=True, parallel = True)
def get_R(L, N, dt):

    R = np.zeros( (N, 2), dtype = np.float32)

    for i in prange(N):
        
        g = np.random.randn(i + 1, 2).astype(np.float32)

        for j in range(i+1): R[i] += L[i, j] @ g[j]

    R *= np.sqrt( 2 * dt )

    return R

def dyadic_product(rij):

    return np.einsum('ij,ik->ijk', rij, rij)

@jit(nopython=True)
def rpy_far(rij_sq, rij, a, viscosity_scale):

    #-------------------------------------------------------------------
    #OUTER PRODUCT
    r        = rij_sq.reshape(-1, 1)
    rij_norm = rij / r
    rij_rij = rij_norm[:, :, None] * rij_norm[:, None, :]

    I = np.eye(2)

    far_field  = ( I + rij_rij ) + (2 * (a/ r)**2).reshape(-1, 1, 1) * ( I / 3 - rij_rij )

    far_field  /= (8 * r ).reshape(-1, 1, 1)

    far_field *= viscosity_scale

    return far_field


@jit(nopython=True)
def rpy_near(rij_sq, rij, a, viscosity_scale):
    
    #-------------------------------------------------------------------
    #OUTER PRODUCT
    r        = rij_sq.reshape(-1, 1)
    rij_norm = rij / r

    rij_rij = rij_norm[:, :, None] * rij_norm[:, None, :]
    
    I = np.eye(2)

    ra = (3 * r / a / 32).reshape(-1, 1, 1)
    
    near_field = ( 1 - 3 * ra ) * I + ra * rij_rij
    
    near_field /= (6 * a)
    
    near_field *= viscosity_scale

    return near_field




