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

test1 = ( np.array([[ 5.0,  5.0]]), np.array([[ 4.0,  5.0]]), np.array([[ 1., 0.]]), ([0,1],[10.,10.]), np.array([[ 6.,  5.]]), np.array([[ 1.,   0.]]) )
test2 = ( np.array([[10.0, 10.0]]), np.array([[ 6.0, 12.0]]), np.array([[ 1., 0.]]), ([0,1],[50.,50.]), np.array([[14., 12.]]), np.array([[ 4.,   2.]]) )
test3 = ( np.array([[10.0, 10.0]]), np.array([[12.0,  9.0]]), np.array([[ 1., 0.]]), ([0,1],[50.,50.]), np.array([[ 8.,  9.]]), np.array([[-2.,  -1.]]) )
test4 = ( np.array([[ 7.0,  8.0]]), np.array([[ 7.5,  6.0]]), np.array([[ np.sqrt(10)/10., 3*np.sqrt(10)/10.]]), ([0,1],[50.,50.]), np.array([[8.6, 9.3]]), np.array([[ 1.6,  1.3]]) )
test5 = ( np.array([[ 7.0,  8.0]]), np.array([[ 6.5, 10.5]]), np.array([[ np.sqrt(10)/10., 3*np.sqrt(10)/10.]]), ([0,1],[50.,50.]), np.array([[5.1, 6.3]]), np.array([[-1.9, -1.7]]) )
test6 = ( np.array([[ 7.0,  8.0]]), np.array([[ 7.5,  6.0]]), np.array([[ np.sqrt(10)/10., 3*np.sqrt(10)/10.]]), ([0,1],[ 8., 9.]), np.array([[0.6, 0.3]]), np.array([[ 1.6,  1.3]]) )

@pytest.mark.parametrize("intersection, p_in, n, pbc_dim, ref_true, ref_vector_true", [test1, test2, test3, test4, test5, test6])
def test_simple_calc_reflection(intersection, p_in, n, pbc_dim, ref_true, ref_vector_true):

    ref, ref_vector = reflection.calc_reflection(intersection = intersection,
                                                 p_in         = p_in,
                                                 n            = n,
                                                 pbc_dim      = pbc_dim)
    
    np.testing.assert_allclose(ref_vector, ref_vector_true)
    np.testing.assert_allclose(ref, ref_true)

    #----------------------------------------------------
    #Test reflection
    pos_p_prev = -1 * ( p_in - intersection )
    pos_p_prev = utils.apply_pbc_vector(vec = pos_p_prev, pbc_dim = pbc_dim)
    pos_ref    = utils.apply_pbc_vector(vec = ref_vector, pbc_dim = pbc_dim)

    dot_pp = np.sum(pos_p_prev  * n, axis = 1)
    dot_pr = np.sum(pos_ref     * n, axis = 1)

    dot_pp /= np.linalg.norm(pos_p_prev, axis = 1)
    dot_pr /= np.linalg.norm(pos_ref   , axis = 1)

    np.testing.assert_allclose(dot_pp, dot_pr)

#@pytest.mark.parametrize("m", [ (np.array([50, 50])), (np.array([65.23, 40.231])), (np.array([12.343, 78.4321])),
#                                (np.array([0, 0])), (np.array([0, 100])), (np.array([100, 0])), (np.array([100, 100])),
#                                (np.array([0, 50])), (np.array([50, 0])), (np.array([100, 50])), (np.array([50, 100]))  ] )
@pytest.mark.parametrize("m", [ (np.array([125, 125])), (np.array([65.23, 40.231])), (np.array([12.343, 78.4321])),
                                (np.array([0, 0])),   (np.array([0, 250])), (np.array([250, 0])),   (np.array([250, 250])),
                                (np.array([0, 125])), (np.array([125, 0])), (np.array([250, 125])), (np.array([125, 250]))  ] )
def test_heavy_calc_reflection_out(m):

    #----------------------------------------------------
    #Set
    N = 100000
    theta = np.linspace(0, 2*np.pi, N)[1:]
    r = 35
    m = m.reshape(1, 2)
    pbc_dim = ([0, 1], [250, 250])
    x_circ = r * np.cos(theta) + m[:, 0]
    y_circ = r * np.sin(theta) + m[:, 1]
    circ   = np.vstack((x_circ, y_circ)).T

    norm   = m - circ

    in_circ = circ + ( 0.5  * np.random.rand(N-1, 1)+.30) * norm + np.random.randn(N-1, 1)

    check_dist2mid = utils.apply_pbc_vector(vec = (in_circ - m), pbc_dim = pbc_dim)
    check_dist2mid = np.linalg.norm( check_dist2mid, axis = 1) < (r - 5)
    np.testing.assert_allclose(check_dist2mid, np.repeat(True, N-1))

    d = in_circ - circ
    #----------------------------------------------------
    #Test reflection
    #d_norm = utils.apply_pbc_vector(vec = np.copy(d), pbc_dim = pbc_dim)
    d_norm = np.linalg.norm(d, axis = 1)
    
    #np.testing.assert_allclose(d_norm < r, np.repeat(True, N-1))
    
    #----------------------------------------------------
    #Calculate normals and intersection
    norm, intersection = reflection.get_normals_circle(prev_points = circ,
                                                       points      = in_circ, 
                                                       d           = d,
                                                       mid         = m,
                                                       r           = r - 5,
                                                       pbc_dim     = pbc_dim)


    circ2intersection = intersection - circ
    circ2intersection = utils.apply_pbc_vector(vec = circ2intersection, pbc_dim = pbc_dim)
    circ2intersection = np.linalg.norm( circ2intersection, axis = 1)
    
    #----------------------------------------------------
    #Calculate reflection
    ref, ref_vector = reflection.calc_reflection(intersection = intersection,
                                                 p_in         = in_circ,
                                                 n            = norm,
                                                 pbc_dim      = pbc_dim)
    

    #Test if length of displace vector is preserved
    d_norm_test = circ2intersection + np.linalg.norm( utils.apply_pbc_vector(vec=ref_vector,pbc_dim=pbc_dim), axis = 1)
 
    np.testing.assert_allclose( d_norm_test, d_norm )

    #Test if all points are in circle
    dist2mid      = utils.apply_pbc_vector( vec = (ref - m), pbc_dim = pbc_dim )
    dist2mid      = np.linalg.norm(dist2mid, axis = 1)
    dist2mid_bool = dist2mid > (r - 5)

    np.save(arr=circ[~dist2mid_bool], file = "failcirc")
    np.save(arr=in_circ[~dist2mid_bool], file="failincirc")
    np.save(arr=ref[~dist2mid_bool], file = "failref")
    np.save(arr=intersection[~dist2mid_bool], file = "failinter")
    """   
    np.save(arr=circ, file = "failcirc")
    np.save(arr=in_circ, file="failincirc")
    np.save(arr=ref, file = "failref")
    np.save(arr=intersection, file = "failinter")
    """
    np.testing.assert_allclose(dist2mid_bool, np.repeat(True, N-1))
    
    #----------------------------------------------------
    #Test reflection
    pos_p_prev = -1 * ( in_circ - intersection )
    pos_p_prev = utils.apply_pbc_vector(vec = pos_p_prev, pbc_dim = pbc_dim)
    pos_ref    = utils.apply_pbc_vector(vec = ref_vector, pbc_dim = pbc_dim)

    dot_pp = np.sum(pos_p_prev  * norm, axis = 1)
    dot_pr = np.sum(pos_ref     * norm, axis = 1)

    dot_pp /= np.linalg.norm(pos_p_prev, axis = 1)
    dot_pr /= np.linalg.norm(pos_ref   , axis = 1)

    np.testing.assert_allclose(dot_pp, dot_pr)

@pytest.mark.parametrize("m", [ (np.array([125, 125])), (np.array([65.23, 40.231])), (np.array([12.343, 78.4321])),
                                (np.array([0, 0])),   (np.array([0, 250])), (np.array([250, 0])),   (np.array([250, 250])),
                                (np.array([0, 125])), (np.array([125, 0])), (np.array([250, 125])), (np.array([125, 250]))  ] )
def test_heavy_calc_reflection_in(m):

    #----------------------------------------------------
    #Set
    N = 100000
    theta = np.linspace(0, 2*np.pi, N)[1:]
    r = 35
    m = m.reshape(1, 2)
    pbc_dim = ([0, 1], [250, 250])
    x_circ = r * np.cos(theta) + m[:, 0]
    y_circ = r * np.sin(theta) + m[:, 1]
    circ   = np.vstack((x_circ, y_circ)).T

    norm   = m - circ

    in_circ = circ + ( 0.5  * np.random.rand(N-1, 1)+.30) * norm + np.random.randn(N-1, 1)

    check_dist2mid = utils.apply_pbc_vector(vec = (in_circ - m), pbc_dim = pbc_dim)
    check_dist2mid = np.linalg.norm( check_dist2mid, axis = 1) < (r - 5)
    np.testing.assert_allclose(check_dist2mid, np.repeat(True, N-1))

    d = circ - in_circ
    #----------------------------------------------------
    #Test reflection
    #d_norm = utils.apply_pbc_vector(vec = np.copy(d), pbc_dim = pbc_dim)
    d_norm = np.linalg.norm(d, axis = 1)
    
    #np.testing.assert_allclose(d_norm < r, np.repeat(True, N-1))
    
    #----------------------------------------------------
    #Calculate normals and intersection
    norm, intersection = reflection.get_normals_circle(prev_points = in_circ,
                                                       points      = circ, 
                                                       d           = d,
                                                       mid         = m,
                                                       r           = r - 5,
                                                       pbc_dim     = pbc_dim)


    circ2intersection = intersection - in_circ
    circ2intersection = utils.apply_pbc_vector(vec = circ2intersection, pbc_dim = pbc_dim)
    circ2intersection = np.linalg.norm( circ2intersection, axis = 1)
    
    #----------------------------------------------------
    #Calculate reflection
    ref, ref_vector = reflection.calc_reflection(intersection = intersection,
                                                 p_in         = circ,
                                                 n            = norm,
                                                 pbc_dim      = pbc_dim)
    

    #Test if length of displace vector is preserved
    d_norm_test = circ2intersection + np.linalg.norm( utils.apply_pbc_vector(vec=ref_vector,pbc_dim=pbc_dim), axis = 1)
 
    np.testing.assert_allclose( d_norm_test, d_norm )

    #Test if all points are in circle
    dist2mid      = utils.apply_pbc_vector( vec = (ref - m), pbc_dim = pbc_dim )
    dist2mid      = np.linalg.norm(dist2mid, axis = 1)
    dist2mid_bool = dist2mid < (r - 5)

    np.save(arr=circ[~dist2mid_bool], file = "failcirc")
    np.save(arr=in_circ[~dist2mid_bool], file="failincirc")
    np.save(arr=ref[~dist2mid_bool], file = "failref")
    np.save(arr=intersection[~dist2mid_bool], file = "failinter")
    """   
    np.save(arr=circ, file = "failcirc")
    np.save(arr=in_circ, file="failincirc")
    np.save(arr=ref, file = "failref")
    np.save(arr=intersection, file = "failinter")
    """
    np.testing.assert_allclose(dist2mid_bool, np.repeat(True, N-1))
    
    #----------------------------------------------------
    #Test reflection
    pos_p_prev = -1 * ( circ - intersection )
    pos_p_prev = utils.apply_pbc_vector(vec = pos_p_prev, pbc_dim = pbc_dim)
    pos_ref    = utils.apply_pbc_vector(vec = ref_vector, pbc_dim = pbc_dim)

    dot_pp = np.sum(pos_p_prev  * norm, axis = 1)
    dot_pr = np.sum(pos_ref     * norm, axis = 1)

    dot_pp /= np.linalg.norm(pos_p_prev, axis = 1)
    dot_pr /= np.linalg.norm(pos_ref   , axis = 1)

    np.testing.assert_allclose(dot_pp, dot_pr)


#---------------------------------------------------------------------
#Test 1
intersection= np.load("../data/tests/test_full_reflection/trajectory_debug_intersection_test1.npy")
prev_points = np.load("../data/tests/test_full_reflection/trajectory_debug_p_older_test1.npy")
points      = np.load("../data/tests/test_full_reflection/trajectory_debug_p_old_test1.npy")
refl        = np.load("../data/tests/test_full_reflection/trajectory_debug_p_test1.npy")

L           = 10
mid         = np.array([[L/2, L/2]])
r           = np.sqrt( (L**2 * 0.3)/2 / np.pi )
pbc_dim     = ([0, 1], [L, L])

prev_index  = np.array([0, 1])
index       = np.array([0, 1])
in_domains  = np.array([True, True])
true_in_domains  = np.array([False, True])

test1 = (mid, r, prev_index, points, prev_points, index, in_domains, pbc_dim, true_in_domains)

@pytest.mark.parametrize("mid, r, prev_index, points, prev_points, index, in_domains, pbc_dim, true_in_domains", [test1])
def test_reflection_workflow(mid, r, prev_index, points, prev_points, index, in_domains, pbc_dim, true_in_domains):

    r_sq = r ** 2

    displace = points - prev_points

    #--------------------------------------------------------------------------------
    org_index  = utils.check_circ_cond(pos = points, mid = mid, r = r_sq, pbc_dim = pbc_dim)

    #This is a correct, but slow, way to get the indices of particles that were in the domain in the previous frame
    prev_index_old = utils.check_circ_cond(pos = prev_points, mid = mid, r = r_sq, pbc_dim = pbc_dim)

    assert list(prev_index) == list(prev_index_old), f'Problem with new list {prev_index} and old list {prev_index_old}'

    #index is a list of particle indices that are reflected by the boundary

    #--------------------------------------------------------------------------------
    #Check if particles are reflected
    if not index.size > 0:
        #If no particles are reflected, the indices of the particles in the domains are passed on
        prev_index = org_index
        np.testing.allclose(in_domains, true_in_domains)
        return 0

    #--------------------------------------------------------------------------------
    #Get indices of reflected particles that remain...
    index_in_domains  = index[  in_domains[index] ] #...inside domains
    index_out_domains = index[ ~in_domains[index] ] #...outside domains

    #--------------------------------------------------------------------------------
    #Get coordinates of reflected particles
    pos_index         = points[index]
    prev_pos_index    = prev_points[index]
    displace_index    = displace[index]

    #--------------------------------------------------------------------------------
    #Calculate normals and intersection with the circular boundary
    norm, intersection    = reflection.get_normals_circle(points      = pos_index,
                                                          prev_points = prev_pos_index,
                                                          d           = displace_index,
                                                          mid         = mid,
                                                          r           = r,
                                                          pbc_dim     = pbc_dim)

    #Calculate new position after reflection
    new_pos, new_displace = reflection.calc_reflection(intersection = intersection, p_in = pos_index, n = norm, pbc_dim = pbc_dim )

    #--------------------------------------------------------------------------------
    #Update coordinates
    points[index]      = new_pos

    #--------------------------------------------------------------------------------
    #There are rare (!) cases in which the particle is placed inside/outside the domain after the reflection
    #The following lines handle with such edge cases

    #Identify misplaced particles
    real_index_in_domains  = index[     utils.check_circ_cond(pos = new_pos, mid = mid, r = r_sq, pbc_dim = pbc_dim) ]
    real_index_out_domains = np.setdiff1d( ar1 = index, ar2 = real_index_in_domains, assume_unique = True)

    #Ideally escaped and captured would be empty
    escaped  = np.setdiff1d(ar1 = index_in_domains,  ar2 =  real_index_in_domains, assume_unique = True)
    captured = np.setdiff1d(ar1 = index_out_domains, ar2 = real_index_out_domains, assume_unique = True)
    

    #The misplaced particles are just re-assigned
    in_domains[ escaped  ] = False
    in_domains[ captured ] = True

    not_reflected_index = np.setdiff1d( ar1 = org_index, ar2 = index, assume_unique =True)
    prev_index =  np.union1d(ar1 = not_reflected_index, ar2 = real_index_in_domains)

    np.testing.assert_equal(in_domains, true_in_domains)
