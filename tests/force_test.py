import numpy as np
import pytest
import matplotlib.pyplot as plt

import brownlipid
from brownlipid import force
from brownlipid import utils

#------------------------------------------------Test generate_pairlist------------------------------------------------#

dist_mat = np.array([ 
                     [np.inf, 0.1111,  5.123, 0.9234],
                     [np.inf, np.inf,  0.342,10.984 ],
                     [np.inf, np.inf, np.inf,  1.1  ],
                     [np.inf, np.inf, np.inf, np.inf],
                     ])

vec_mat  = np.array([
                     [[np.inf, np.inf], [0.3434, 0.2432], [0.4242,95.5765], [7.4352, 2.2455]],
                     [[np.inf, np.inf], [np.inf, np.inf], [0.4353,86.9763], [6.7655, 3.9988]],
                     [[np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf], [5.6543,20.4546]],
                     [[np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf]],
                     ])

N = 4

lj_buffer = 1.1

rij_true    = np.array([[0.3434, 0.2432], [7.4352, 2.2455], [0.4353,86.9763], [5.6543,20.4546]])
rij_sq_true = np.array([0.1111, 0.9234, 0.342, 1.1])

pairlist_true = np.array([
                          [0, 1],
                          [0, 3],
                          [1, 2],
                          [2, 3]])

outer_rij_true = np.array([[0.4242,95.5765], [6.7655, 3.9988]])
outer_rij_sq_true = np.array([5.123, 10.984])

outer_pairlist_true = np.array([
                                [0, 2],
                                [1, 3]])

offsets = np.zeros((N,N), dtype = np.float32)

test1 = (dist_mat, vec_mat, N, lj_buffer, rij_true, rij_sq_true, offsets, pairlist_true, outer_rij_true, outer_rij_sq_true, outer_pairlist_true)

####################

dist_mat = np.array([ 
                     [np.inf, 0.1811, 15.123, 0.9234, 0.34342],
                     [np.inf, np.inf,100.342,10.0343,33.34343],
                     [np.inf, np.inf, np.inf,  1.1  ,0.000001],
                     [np.inf, np.inf, np.inf, np.inf, 10.0001],
                     [np.inf, np.inf, np.inf, np.inf,  np.inf]
                     ])

vec_mat  = np.array([
                     [[np.inf, np.inf], [0.3434, 0.2432], [0.4242,95.5765], [7.4352, 2.2455], [5.654,20.4546]],
                     [[np.inf, np.inf], [np.inf, np.inf], [0.4353,86.9763], [6.7655, 3.9988], [5.6023,30.4546]],
                     [[np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf], [5.6543,20.4546], [5.6533, 1.4546]],
                     [[np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf], [5.6493, 2.4546]],
                     [[np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf]],
                     ])

N = 5

lj_buffer = 1.1

rij_true    = np.array([[0.3434, 0.2432], [7.4352, 2.2455], [5.654,20.4546], [5.6543,20.4546], [5.6533, 1.4546]])
rij_sq_true = np.array([0.1811, 0.9234, 0.34342, 1.1, 0.000001])

pairlist_true = np.array([
                          [0, 1],
                          [0, 3],
                          [0, 4],
                          [2, 3],
                          [2, 4]])

outer_rij_true = np.array([[0.4242,95.5765], [0.4353,86.9763], [6.7655, 3.9988], [5.6023,30.4546], [5.6493, 2.4546]])
outer_rij_sq_true = np.array([15.123, 100.342,10.0343,33.34343, 10.0001])

outer_pairlist_true = np.array([
                                [0, 2],
                                [1, 2],
                                [1, 3],
                                [1, 4],
                                [3, 4]])

offsets = np.zeros((N,N), dtype = np.float32)

test2 = (dist_mat, vec_mat, N, lj_buffer, rij_true, rij_sq_true, offsets, pairlist_true, outer_rij_true, outer_rij_sq_true, outer_pairlist_true)

@pytest.mark.parametrize("dist_mat, vec_mat, N, lj_buffer, rij_true, rij_sq_true, offsets, pairlist_true, outer_rij_true, outer_rij_sq_true, outer_pairlist_true", [test1, test2])
def test_generate_pairlist(dist_mat, vec_mat, N, lj_buffer, rij_true, rij_sq_true, offsets, pairlist_true, outer_rij_true, outer_rij_sq_true, outer_pairlist_true):

    rij, rij_sq, pairlist, outer_rij, outer_rij_sq, outer_pairlist = force.generate_pairlist(dist_mat = dist_mat, vec_mat = vec_mat, lj_buffer = lj_buffer)

    np.testing.assert_allclose(rij,    rij_true)
    np.testing.assert_allclose(rij_sq, rij_sq_true)
    np.testing.assert_allclose(pairlist, pairlist_true)
    np.testing.assert_allclose(outer_rij, outer_rij_true)
    np.testing.assert_allclose(outer_rij_sq, outer_rij_sq_true)
    np.testing.assert_allclose(outer_pairlist, outer_pairlist_true)

def test_update_pairlist_no_pbc():
    """
    Test update_pairlist with no periodic boundary conditions.
    Verify correct distance vector and squared distance calculation.
    """
    # Create sample positions and pairlist
    pos = np.array([
        [0.0, 0.0],   # Particle 0
        [1.0, 1.0],   # Particle 1
        [3.0, 3.0]    # Particle 2
    ])

    # Pairlist with pairs (0,1) and (0,2)
    pairlist = np.array([
        [0, 1],  # Pair of particle indices
        [0, 2]
    ])

    # Periodic boundary conditions
    pbc_dim = ([0, 1], [1000, 1000])

    offsets = np.zeros((3, 3), dtype = np.float32)

    # Call the function
    rij, rij_sq = force.update_pairlist(pos, pos, pairlist, pbc_dim, offsets[pairlist[:, 0], pairlist[:, 1]])

    # Expected distance vectors
    expected_rij = np.array([
        [-1.0, -1.0],   # Distance vector from particle 1 to particle 0
        [-3.0, -3.0]    # Distance vector from particle 2 to particle 0
    ])

    # Expected squared distances
    expected_rij_sq = np.sqrt( np.array([2.0, 18.0]) )

    # Assertions
    np.testing.assert_array_almost_equal(rij, expected_rij)
    np.testing.assert_array_almost_equal(rij_sq, expected_rij_sq)

def test_update_pairlist_with_pbc():
    """
    Test update_pairlist with periodic boundary conditions.
    Verify correct handling of distances that wrap around the periodic box.
    """
    # Create sample positions in a periodic box
    pos = np.array([
        [0.1, 0.1],   # Particle 0
        [9.9, 9.9],   # Particle 1 (near opposite edge)
    ])
    
    # Pairlist with pair (0,1)
    pairlist = np.array([
        [0, 1]
    ])
    
    # Periodic boundary conditions on both x and y axes with box size 10
    pbc_dim = ([0, 1], [10.0, 10.0])
    
    offsets = np.zeros((3, 3), dtype = np.float32)
    
    # Call the function
    rij, rij_sq = force.update_pairlist(pos, pos, pairlist, pbc_dim, offsets[pairlist[:, 0], pairlist[:, 1]])
    
    # Expected distance vector (wrapped around)
    # Distance from (9.9, 9.9) to (0.1, 0.1)
    expected_rij = np.array([
        [0.2, 0.2]
    ])
    
    # Expected squared distance
    expected_rij_sq = np.sqrt( np.array([0.08]) )
    
    # Assertions
    np.testing.assert_array_almost_equal(rij, expected_rij)
    np.testing.assert_array_almost_equal(rij_sq, expected_rij_sq)

def test_update_pairlist_input_validation():
    """
    Test input validation and error handling scenarios.
    """
    # Positions array
    pos = np.array([
        [0.0, 0.0],
        [1.0, 1.0]
    ])
    
    # Test with invalid pairlist
    with pytest.raises(IndexError):
        force.update_pairlist(pos, pos, np.array([[10, 1]]), ([1], [23]), np.zeros(2) )
    
    # Test with mismatched PBC dimensions
    force.update_pairlist(pos, pos, np.array([[0, 1]]), ([0], [10.0, 20.0]), np.zeros(2))

def test_calculate_force_basic_symmetry():
    """
    Test basic force calculation symmetry and Newton's third law.
    Verify that forces between two particles are equal and opposite.
    """
    # Simulate two particles
    N = 2

    # Distance vector from particle 1 to particle 0
    rij = np.array([[1.34, 1.4]])  # Particle 1 to Particle 0

    # Squared distance
    rij_sq = np.linalg.norm(rij, axis = 1)**2

    # Masked pairlist (indices of interacting particles)
    masked_pairlist = np.array([[0, 1]])

    # Lennard-Jones parameters (typical values)
    lj_A12 = 1.0
    lj_B6 = 1.0

    # Calculate forces
    force_per_particle = force.calculate_force(rij, rij_sq, N, lj_A12, lj_B6, masked_pairlist)

    # Check force magnitude is equal
    np.testing.assert_almost_equal(
        np.linalg.norm(force_per_particle[0]),
        np.linalg.norm(force_per_particle[1])
    )

    # Check forces are opposite
    np.testing.assert_almost_equal(force_per_particle[0], -force_per_particle[1])

def test_calculate_force_no_interaction():
    """
    Test force calculation when no particles are close enough to interact.
    """
    # Simulate particles far apart
    N = 2

    # Large distance vector
    rij = np.array([[10.0, 0.0]])

    # Large squared distance
    rij_sq = np.array([100.0])

    # Masked pairlist (indices of interacting particles)
    masked_pairlist = np.array([[0, 1]])

    # Lennard-Jones parameters
    lj_A12 = 1.0
    lj_B6 = 1.0

    # Calculate forces
    force_per_particle = force.calculate_force(rij, rij_sq, N, lj_A12, lj_B6, masked_pairlist)

    # Check forces are essentially zero
    np.testing.assert_array_almost_equal(force_per_particle, np.zeros((N, 2)))

def test_calculate_force_multiple_pairs():
    """
    Test force calculation with multiple particle pairs.
    """
    # More complex scenario with multiple interacting pairs
    N = 4
    
    # Distance vectors
    rij = np.array([
        [1.0, 0.0],   # Pair 0-1
        [0.0, 1.0],   # Pair 2-3
    ])
    
    # Squared distances
    rij_sq = np.array([1.0, 1.0])
    
    # Masked pairlist (multiple pairs)
    masked_pairlist = np.array([
        [0, 1],  # First pair
        [2, 3]   # Second pair
    ])
    
    # Lennard-Jones parameters
    lj_A12 = 1.0
    lj_B6 = 1.0
    
    # Calculate forces
    force_per_particle = force.calculate_force(rij, rij_sq, N, lj_A12, lj_B6, masked_pairlist)
    
    # Verify total number of particles
    assert force_per_particle.shape == (N, 2)

def test_calculate_force_values():
    
    def naive_force(r, sig, eps): return 48 * eps * ((sig/r)**12 - 0.5 * (sig/r)**6) / r

    lj_sig = 0.7706
    lj_eps = 0.3221

    lj_A12 = 48 * lj_eps * lj_sig**12
    lj_B6  = 24 * lj_eps * lj_sig**6

    rij      = 1.1*np.random.rand(1000000,2).astype(np.float32) + 0.8
    rij_norm = 1.1*np.random.rand(1000000,1).astype(np.float32) + 0.8
    rij_norm = np.sort(rij_norm)

    rij      = rij_norm * rij / np.linalg.norm(rij, axis = 1).reshape(-1,1)
    rij_sq    = rij_norm**2

    y_pred = force.calculate_force(rij, rij_sq, N = int(2E6), lj_A12 = lj_A12, lj_B6= lj_B6, masked_pairlist=np.split(np.arange(int(2E6)),int(1E6)))
    y_true = naive_force(r = rij_norm, sig = lj_sig, eps = lj_eps) * rij/rij_norm
    y_true = y_true.astype(np.float32)

    np.testing.assert_allclose(np.abs(y_pred[::2]), np.abs(y_true), atol=1E-5, rtol=0)

    r = np.linspace(0.1, 5, 50001)
    plt.scatter(rij_norm, (y_true/rij*rij_norm)[:, 1], s=50, marker='o', facecolor='None', edgecolor = 'blue')
    plt.scatter(rij_norm, (y_pred[::2]/rij*rij_norm)[:, 1], s=10, marker = "x", lw=3, color = 'r')
    plt.plot(r, naive_force(r, lj_sig, lj_eps), color = 'green')
    plt.ylim(-2, 2)
    plt.xlim(0, 2)

    plt.savefig("test_force_kernel.png", dpi = 300)


#----------------------------------------------------------------------------------------------------------------------------------------------------------

#@pytest.mark.parametrize("L, apl", [(5, 0.66), (5, 2.35045)])
@pytest.mark.parametrize("L, apl", [(10, 0.66), (10, 2.35045)])
def test_pair_function_no_domains(L, apl):

    d_coeff = 0.0000504694
    nsteps  = 1E7
    dt      = 0.1
    N_Lipids = int( np.round( (L**2) / apl ) )

    uni = brownlipid.Universe(
                          size_x = L,
                          size_y = L,
                               N = N_Lipids,
                         pbc_dim = 'xy',
                          nsteps = int(nsteps),
                              dt = dt,
                         nstxout = 100,
                    base_d_coeff = d_coeff,
                 external_forces = {'epsilon': 0.3221, 'sigma': 0.7706, 'r_vdw': 1.2, 'r_list':3.0, 'nstlist':10},
                         output  = f"no_domains/trajectory_pp_g_{L}_{apl}",
                         nstchk  = int(nsteps) )

    uni.evolve()

@pytest.mark.parametrize("L, r,apl", [(10, 4, 0.66), (10, 2.5, 0.66), (10, 2.5, 2.35045)])
def test_pair_function_domains(L, r, apl):

    d_coeff = 0.0000504694
    nsteps  = 1E7
    dt      = 0.1

    effA = L**2 - np.pi * r**2

    N_Lipids = int( np.round( effA / apl ) )

    uni = brownlipid.Universe(
                          size_x = L,
                          size_y = L,
                               N = N_Lipids,
                         pbc_dim = 'xy',
                          nsteps = int(nsteps),
                 hard_boundaries = {"Circle":[[np.inf, r, L/2, L/2]]},
                        softwall = True,
                              dt = dt,
                         nstxout = 100,
                    base_d_coeff = d_coeff,
                 external_forces = {'epsilon': 0.3221, 'sigma': 0.7706, 'r_vdw': 1.2, 'r_list':3.0, 'nstlist':10},
                         output  = f"access_domain/trajectory_L_{L}_APL_{apl}_r_{r}",
                         nstchk  = int(nsteps) )

    uni.evolve()
