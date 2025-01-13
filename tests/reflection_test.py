import numpy as np
import pytest

from brownlipid import reflection
from brownlipid import utils

#-------------------------------------------------------------------------
prev_points = np.array([[ 13.00,  6.00]], dtype = np.float32)
points      = np.array([[  6.00,  6.00]], dtype = np.float32)
d           = np.array([[- 7.00,  0.00]], dtype = np.float32)
mid         = np.array([[  5.00,  6.00]], dtype = np.float32)
r           = 3.0
pbc_dim     = ([0, 1], [100., 100.])
inter_true  = np.array([[  8.00,  6.00]], dtype = np.float32)
norm_true   = np.array([[  1.00,  0.00]], dtype = np.float32)

test1 = (prev_points, points, d, mid, r, pbc_dim, inter_true, norm_true)
del prev_points, points, d, mid, r, pbc_dim, inter_true, norm_true

#-------------------------------------------------------------------------
prev_points = np.array([[  4.50, -0.50]], dtype = np.float32)
points      = np.array([[  4.00,  1.00]], dtype = np.float32)
d           = np.array([[ -0.50,  1.50]], dtype = np.float32)

mid         = np.array([[  3.50,  1.50]], dtype = np.float32)
pbc_dim     = ([0, 1], [5., 5.])
r           = 1.50

A           =  2.5
B           = -3.5
C           =  5.0

inter_true  = prev_points + ( - (B/A) - ( (B/A)**2 - (C - r**2) /A )**0.5  ) * d
norm_true   = inter_true - mid
norm_true   /= np.linalg.norm(norm_true)

test2 = (prev_points, points, d, mid, r, pbc_dim, inter_true, norm_true)
del prev_points, points

prev_points = np.array([[  4.50,  4.50]], dtype = np.float32)
points      = np.array([[  4.00,  1.00]], dtype = np.float32)

test3 = (prev_points, points, d, mid, r, pbc_dim, inter_true, norm_true)
del prev_points, points

prev_points = np.array([[  4.50,  4.50]], dtype = np.float32)
points      = np.array([[  4.00,  6.00]], dtype = np.float32)

test4 = (prev_points, points, d, mid, r, pbc_dim, inter_true, norm_true)
del prev_points, points, d, mid, r, pbc_dim, inter_true, norm_true

@pytest.mark.parametrize("prev_points,points,d,mid,r,pbc_dim,inter_true,norm_true", [test1, test2, test3, test4])
def test_get_normals_circle(prev_points, points, d, mid, r, pbc_dim, inter_true, norm_true):

    norm, intersection = reflection.get_normals_circle(prev_points = prev_points,
                                                       points      = points, 
                                                       d           = d,
                                                       mid         = mid,
                                                       r           = r,
                                                       pbc_dim     = pbc_dim)

    for k, size in zip(pbc_dim[0], pbc_dim[1]): 
        intersection[:, k] %= size

    norm = utils.apply_pbc_vector( vec = norm, pbc_dim = pbc_dim)

    np.testing.assert_allclose(intersection, inter_true)
    np.testing.assert_allclose(norm, norm_true)
    
    #-------------------------------------
    #Check if intersection is between old positions and new positions
    #The following code checks if the intersection is located on the shortest vector between the previous and the
    #current position based on the smallest image convention.
    
    check, d2 = utils.test_intersection(intersection = intersection,
                                        curr_points  = points,
                                        prev_points  = prev_points,
                                        displace     = d,
                                        pbc_dim      = pbc_dim)

    test_result = np.allclose(check, d2) & np.allclose(d2, check)

    np.testing.assert_equal(test_result, True)



@pytest.mark.parametrize("m", [ (np.array([50, 50])), (np.array([65.23, 40.231])), (np.array([12.343, 78.4321])),
                                (np.array([0, 0])), (np.array([0, 100])), (np.array([100, 0])), (np.array([100, 100])),
                                (np.array([0, 50])), (np.array([50, 0])), (np.array([100, 50])), (np.array([50, 100]))  ] )
def test_heavy_get_normals_circle(m):

    N = 10000
    theta = np.linspace(0, 2*np.pi, N)[1:]
    r = 35
    m = m.reshape(1, 2)
    pbc_dim = ([0, 1], [100, 100])

    x_circ = r * np.cos(theta) + m[:, 0]
    y_circ = r * np.sin(theta) + m[:, 1]

    circ   = np.vstack((x_circ, y_circ)).T

    norm   = m - circ

    in_circ = circ + 0.5 * norm + 1.1 * np.random.randn(N - 1, 2)

    d = in_circ - circ

    norm, intersection = reflection.get_normals_circle(prev_points = circ,
                                                       points      = in_circ, 
                                                       d           = d,
                                                       mid         = m,
                                                       r           = r - 5,
                                                       pbc_dim     = pbc_dim)

    for k, size in zip(pbc_dim[0], pbc_dim[1]): 
        intersection[:, k] %= size

    #-------------------------------------
    #Check if intersection is on circle boundary

    vec2mid = utils.apply_pbc_vector( vec = intersection - m, pbc_dim = pbc_dim )

    dist2mid = np.sqrt( np.sum( vec2mid**2 , axis = 1) )

    np.testing.assert_allclose(dist2mid, np.ones(N - 1) * (r - 5))

    #-------------------------------------
    #Check if intersection is between old positions and new positions
    #The following code checks if the intersection is located on the shortest vector between the previous and the
    #current position based on the smallest image convention.
    check, d2 = utils.test_intersection(intersection = intersection,
                                        curr_points  = in_circ,
                                        prev_points  = circ,
                                        displace     = d,
                                        pbc_dim      = pbc_dim)

    test_result = np.allclose(check, d2) & np.allclose(d2, check)

    np.testing.assert_equal(test_result, np.repeat(True, N - 1))

#    if not np.all( np.abs(check - prev_d2) < 1E-8 ):
#
#        print(lam1, lam2)
#        print(lam)
#        print('A',A)
#        print('B',B)
#        print('C',C)
#        print('r', r)
#        print('mid', mid)
#
#        print("Something went wrong. Write debug files!")
#        np.save(arr = prev_points, file = 'prev_positions.debug.npy')
#        np.save(arr = points,      file = 'positions.debug.npy')
#        np.save(arr = d,           file = 'displace.debug.npy')
#        np.save(arr = d,           file = 'displace.debug.npy')
#
#        raise ValueError('Intersection is not between positions!')
#
