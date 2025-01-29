from brownlipid import utils

import numpy as np

import pytest

@pytest.mark.parametrize("x, threshold", [(5, 6), (6, 6), (7, 6)])
def test_heaviside(x, threshold):


    output = utils.heaviside(x, threshold)

    if x <= threshold: np.testing.assert_allclose(output, 1.0, rtol = 0, atol = 1E-5)
    else: np.testing.assert_allclose(output, 0.0, rtol = 0, atol = 1E-5)


#Distance matrix tests
#------------------------------------------------------------
L    = 4.1
N    = 2
pos  = np.array([[1, 1], [3, 3]], dtype = np.float32)
dist = np.sum( (pos[0] - pos[1])**2 ).astype(np.float32)

dist_mat = np.ones( (N, N), dtype = np.float32 )    * np.inf
vec_mat  = np.ones( (N, N, 2), dtype = np.float32 ) * np.inf

pbc_dim = ([0,1], [L, L])

dist_mat[0, 1] = dist
vec_mat[0, 1] = pos[0] - pos[1]
test1 = (pos, N, ([0,1], (L, L)), dist_mat, vec_mat )
del pos, N, pbc_dim, dist_mat, vec_mat

#------------------------------------------------------------
L    = 10.
N    = 2
pos  = np.array([[1, 5], [9, 5]], dtype = np.float32)

dist_mat = np.ones( (N, N) )    * np.inf
vec_mat  = np.ones( (N, N, 2) ) * np.inf

pbc_dim = ([0,1], [L, L])
dist_mat[0, 1] = 2**2
vec_mat[0, 1]  = np.array([2, 0])
test2 = (pos, N, pbc_dim, dist_mat, vec_mat )

del pos, N, pbc_dim, dist_mat, vec_mat
#------------------------------------------------------------
L    = 10.
N    = 2
pos  = np.array([[1, 5], [9, 5]], dtype = np.float32)

dist_mat = np.ones( (N, N) )    * np.inf
vec_mat  = np.ones( (N, N, 2) ) * np.inf

pbc_dim = ([1], [L])
dist_mat[0, 1] = 8**2
vec_mat[0, 1]  = pos[0] - pos[1]
test3 = (pos, N, pbc_dim, dist_mat, vec_mat )
del pos, N, pbc_dim, dist_mat, vec_mat

#------------------------------------------------------------
L    = 10.
N    = 2
pos  = np.array([[4, 1], [4, 9]], dtype = np.float32)

dist_mat = np.ones( (N, N) )    * np.inf
vec_mat  = np.ones( (N, N, 2) ) * np.inf

pbc_dim = ([0,1], [L, L])
dist_mat[0, 1] = 2**2
vec_mat[0, 1]  = np.array([0, 2])
test4 = (pos, N, pbc_dim, dist_mat, vec_mat )
del pos, N, pbc_dim, dist_mat, vec_mat

#------------------------------------------------------------
L    = 10.
N    = 2
pos  = np.array([[4, 1], [4, 9]], dtype = np.float32)

dist_mat = np.ones( (N, N) )    * np.inf
vec_mat  = np.ones( (N, N, 2) ) * np.inf

pbc_dim = ([0], [L])
dist_mat[0, 1] = 8**2
vec_mat[0, 1]  = pos[0] - pos[1]
test5 = (pos, N, pbc_dim, dist_mat, vec_mat )
del pos, N, pbc_dim, dist_mat, vec_mat

#------------------------------------------------------------
L    = 100000.
N    = 1000
pos  = 500 * np.random.rand(N, 2).astype(np.float32)

dist_mat = np.ones( (N, N) )    * np.inf
vec_mat  = np.ones( (N, N, 2) ) * np.inf

pbc_dim = ([0, 1], [L, L])

for i in range(N):
    for j in range(i+1, N):
        vec_mat[i,j] = (pos[i] - pos[j]).astype(np.float32)
        dist_mat[i,j] = np.sum(vec_mat[i,j]**2)

test6 = (pos, N, pbc_dim, dist_mat, vec_mat )
del pos, N, pbc_dim, dist_mat, vec_mat

#------------------------------------------------------------
L    = 500.
N    = 1000
pos  = 500 * np.random.rand(N, 2).astype(np.float32)

dist_mat = np.ones( (N, N) )    * np.inf
vec_mat  = np.ones( (N, N, 2) ) * np.inf

pbc_dim = ([0, 1], [L, L])

for i in range(N):
    for j in range(i+1, N):
        vec_mat[i,j] = (pos[i] - pos[j]).astype(np.float32)

        vec_mat[i,j] = utils.apply_pbc_vector(vec = vec_mat[i,j].reshape(1, 2), pbc_dim = pbc_dim)

        dist_mat[i,j] = np.sum(vec_mat[i,j]**2)

test7 = (pos, N, pbc_dim, dist_mat, vec_mat )
del pos, N, pbc_dim, dist_mat, vec_mat


@pytest.mark.parametrize("pos, N, pbc_dim, dist_true, vec_true", [test1, test2, test3, test4, test5, test6, test7])
def test_distance_matrix(pos, N, pbc_dim, dist_true, vec_true):

    dist_out, vec_out = utils.distance_matrix_NxN(pos, N, pbc_dim, np.zeros((N,N), dtype = np.float32))

    np.testing.assert_allclose(dist_out, np.sqrt(dist_true).astype(np.float32), atol=1E-4, rtol=0)
    np.testing.assert_allclose(vec_out.astype(np.float32), vec_true.astype(np.float32), atol=1E-4, rtol=0)

test1=( np.array([[ 1  , 2  ]]), ([0, 1],[5, 5]), np.array([[1,    2]]) )
test2=( np.array([[ 1  , 2  ]]), ([0, 1],[5, 3]), np.array([[1,   -1]]) )
test3=( np.array([[ 1  , 2  ]]), ([0], [5]),      np.array([[1,    2]]) )
test4=( np.array([[ 1  , 2  ]]), ([0], [5]),      np.array([[1,    2]]) )
test4=( np.array([[-2.5, 2.5]]), ([0], [4]),      np.array([[1.5, -1.5]]) )

@pytest.mark.parametrize("vec, pbc_dim, out_true", [test1, test2, test3])
def test_apply_pbc_vector(vec, pbc_dim, out_true):

    out_ = utils.apply_pbc_vector(vec, pbc_dim)

    np.testing.assert_allclose(out_, out_true)

test1=( np.array([[ 7.33, 9.99]]), np.array([[4, 5]]), 3, ([0, 1],[15, 15]), np.array([], dtype = np.int64) )
test2=( np.array([[ 4.10, 2.50]]), np.array([[4, 5]]), 3, ([0, 1],[15, 15]), np.array([0], dtype = np.int64) )
test3=( np.array([[ 4.10, 2.50], [ 7.33, 9.99]]), np.array([[4.0, 5.0]]), 3, ([0, 1],[15, 15]), np.array([0], dtype = np.int64) )
test4=( np.array([[ 7.33, 9.99], [ 4.10, 2.50]]), np.array([[4.0, 5.0]]), 3, ([0, 1],[15, 15]), np.array([1], dtype = np.int64) )
test5=( np.array([[ 2.90, 7.49], [ 0.00, 0.00], [5.00, 8.00]]), np.array([[0.0, 7.5]]), 3, ([0, 1],[7.5, 15.]), np.array([0,2], dtype = np.int64) )
test6=( np.array([[ 2.90, 7.49], [ 0.00, 0.00], [5.00, 8.00]]), np.array([[0.0, 7.5]]), 3, ([1],[15.]), np.array([0], dtype = np.int64) )
test7=( np.array([[ 2.90, 7.49], [ 0.00, 0.00], [5.00, 8.00]]), np.array([[0.0, 7.5]]), 3, ([0],[7.5]), np.array([0, 2], dtype = np.int64) )
test8=( np.array([[ 2.90, 7.49], [ 0.00, 0.00], [5.00, 8.00]]), np.array([[0.0, 7.5]]), 3, ([0, 1],[15., 15.]), np.array([0], dtype = np.int64) )

@pytest.mark.parametrize("pos, mid, r, pbc_dim, out_true", [test1, test2, test3, test4, test5, test6, test7, test8])
def test_check_circ_cond(pos, mid, r, pbc_dim, out_true):
    
    out_ = utils.check_circ_cond(pos, mid, r ** 2, pbc_dim)
    
    np.testing.assert_equal(out_, out_true)


test1=( np.array([5, 5]), np.array([2, 2]), np.array([5.99, 5.99]), np.array([3.99, 3.99]), ([0,1],[10, 10]), True )
test2=( np.array([0, 0]), np.array([2, 2]), np.array([5.99, 5.99]), np.array([2.01, 2.01]), ([0,1],[ 6,  6]), True )
test3=( np.array([5, 5]), np.array([2, 2]), np.array([5.99, 5.99]), np.array([2.01, 2.01]), ([0,1],[ 6,  6]), False )

@pytest.mark.parametrize("intersection, curr_points, prev_points, displace, pbc_dim, result", [test1, test2, test3])
def test_test_intersection(intersection, curr_points, prev_points, displace, pbc_dim, result):


    check, d2 = utils.test_intersection(intersection = intersection,
                                        curr_points  = curr_points,
                                        prev_points  = prev_points,
                                        displace     = displace,
                                        pbc_dim      = pbc_dim)

    test_result = np.allclose(check, d2) & np.allclose(d2, check)

    np.testing.assert_equal(test_result, result)

@pytest.mark.parametrize("F, N", [(1000, 50), (343, 38), (95, 199)])
def test_msd(F, N):

    pos = np.random.rand(F, N, 2) * 100

    lagtimes = np.arange(0, pos.shape[0], 1)

    msd,     sd_per_particle     = utils.evaluate_lagtimes(pos = pos, lagtimes = lagtimes, N = pos.shape[1])
    msd_fft, sd_per_particle_fft = utils.MSD_fft_ax(pos = pos)

    msd_fft[0] = 0

    np.testing.assert_allclose(msd, msd_fft)
    np.testing.assert_allclose(sd_per_particle, sd_per_particle)
