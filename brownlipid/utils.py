# ----PYTHON---- #
"""
utils.py

This module contains a bunch of helper functions that are used in multiple steps of the main workflow.

"""


import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from numba import jit,prange

def load_params(json_file):

    #Read json file
    with open(json_file, 'r') as f:
        input_params = json.load(f)

    #Filter for parameters (important when using run info file as input)
    if any(k == 'parameters' for k in input_params.keys()): input_params = input_params["parameters"]

    #Check for unknown values in parameter file
    unknown = set(input_params) - {'size_x', 'size_y', 'N', 'pbc_dim', 'nsteps', 'dt', 'nstxout', 'nstchk', 'base_d_coeff', 'hard_boundaries', 'softwall', 'bounce_scale', 'checkpoint_structure', 'viscosity', 'random_domains', 'diffusion_domains', 'external_forces', 'temp', 'pressure_coupling', 'output'}
    if unknown: raise KeyError(f"Unknown parameter(s): {unknown}")

    #Define default values
    params = {
        'size_x': 10.0,
        'size_y': 10.0,
        'N': 1000,
        'pbc_dim': 'xy',
        'nsteps': 5000,
        'dt': 1.0,
        'nstxout': 1,
        'nstchk': 1,
        'base_d_coeff': 1.0,
        'hard_boundaries': {},
        'softwall': False,
        'bounce_scale': None,
        'checkpoint_structure': None,
        'viscosity': 1,
        'random_domains': False,
        'diffusion_domains': 0.0,
        'external_forces': {},
        'temp': 298,
        'pressure_coupling': {},
        'output': 'output'
    }

    #Overwrite default values with given values
    params.update(input_params)

    #Make sure that the datatypes are correct
    params['nsteps'] = int(params['nsteps'])
    params['nstxout'] = int(params['nstxout'])
    params['nstchk'] = int(params['nstchk'])

    return params


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

@jit(nopython=True)#, parallel=True)
def distance_matrix_NxN(pos, N, pbc_dim, offsets):

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
    #dist_mat = np.zeros((N, N)   , dtype = np.float32) * np.nan
    #vec_mat  = np.zeros((N, N, 2), dtype = np.float32) * np.nan
    
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
        rij_       = np.sqrt( np.sum(rij**2,axis=1) )
        rij_offset = rij_ - offsets[i, (i+1):]

        #Store the squared distances and distance vectors in the arrays
        dist_mat[i, (i+1):]    = rij_offset
        vec_mat[ i, (i+1):, :] = ( rij_offset / rij_ ).reshape(-1, 1) * rij

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
def apply_pbc_vector_3dim(vec, pbc_dim) -> np.ndarray:

        """
        Periodic boundary conditions for DIRECTIONAL VECTORS a.k.a. VECTORS.

        If a component of a vector is larger than half the box size (positive and negative), subtract one box size.

        """
        
        assert vec.ndim == 3, 'Vector has not a dimension of 2'
        
        #Apply PBC
        for i, size in zip(pbc_dim[0], pbc_dim[1]):
            vec[:, :, i] = np.where(vec[:, :, i] >    size / 2, vec[:, :, i] - size, vec[:, :, i])
            vec[:, :, i] = np.where(vec[:, :, i] <= - size / 2, vec[:, :, i] + size, vec[:, :, i])

        return vec

@jit(nopython=True)
def apply_hard_wall_scale(pos, d, hard_wall, bounce_scale):

    """
    Periodic boundary conditions for POSITIONAL VECTORS a.k.a. POINTS.

    Particles leaving the box on one site, enter the box again from the opposite site.
    Since, a simple rectangular box shape is used as unit cell the modulo operator is applied here.

    """

    if len( hard_wall[0] ) == 0: return pos

    assert pos.ndim == 2, 'Position vector has not a dimension of 2'

    d_norm = bounce_scale * d / np.sqrt( np.sum( d**2, axis = 1) ).reshape(-1,1)


    #Apply periodic boundary conditions for every requested dimension
    for i, size in zip(hard_wall[0], hard_wall[1]):

        changed = np.array([-1 * i + 1 * (1-i), 1 * i - 1 * (1-i)], dtype = np.float32)

        mask0 = pos[:, i] < 0
        maskL = pos[:, i] > size

        lamd0 =        (pos[mask0, i]  / d[mask0, i]).reshape(-1,1) * d[mask0]
        lamdL = ((-size+pos[maskL, i]) / d[maskL, i]).reshape(-1,1) * d[maskL]

        pos[mask0] -= lamd0 + np.sqrt(np.sum( lamd0**2, axis = 1)).reshape(-1, 1) * (changed * d_norm[mask0])

        pos[maskL] -= lamdL + np.sqrt(np.sum( lamdL**2, axis = 1)).reshape(-1, 1) * (changed * d_norm[maskL])


    return pos

@jit(nopython=True)
def apply_soft_wall_force(pos, hard_wall, lj_cutoff, lj_A12, lj_B6):

    """
    Walls interact via Lennard-Jones interactions with particles.
    Force is only acting on particles, walls have "infinite" mass.

    pos := numpy.ndarray
        Position of particles. Shape is (N, 2)
    hard_wall := tuple
        Information about which box vectors are hard walls. ([INDEX], [SIZE])
    lj_cutoff := float
        Cutoff for Lennard-Jones interactions. Particles with larger distance than lj_cutoff are not taken into account.
    lj_A12 := float
        Parameter for Lennard-Jones potential.
    lj_B6 := float
        Parameter for Lennard-Jones potential.

    """

    force_from_wall = np.zeros_like(pos, dtype=np.float32)

    if not hard_wall[0]: return force_from_wall

    for i, size in zip(hard_wall[0], hard_wall[1]):
        
        # Distances from walls
        ri = np.where(pos[:, i] > (size / 2), pos[:, i] - size, pos[:, i])

        # Squared distances
        ri_sq = ri ** 2

        # Mask for particles within cutoff
        lj_mask = (ri_sq <= lj_cutoff)

        if not np.any(lj_mask): continue

        # Lennard-Jones force calculation
        inv_ri_sq = 1.0 / ri_sq[lj_mask]
        
        sr6 = inv_ri_sq ** 3
        sr12 = sr6 ** 2
        force = (lj_A12 * sr12 - lj_B6 * sr6) * inv_ri_sq

        force_from_wall[lj_mask, i] += force * ri[lj_mask]

    return force_from_wall

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

def get_elements_only_in_a(a, b, assume_unique):

    """
    Obtain set difference between a and b.

    a := numpy.ndarray
        array to test against
    b := numpy.ndarray
        array containing elements that are tested
    assume_unique := boolean

    """

    #mask has the shape of a
    mask = ~np.isin( element = a, test_elements = b, assume_unique = assume_unique )
    #   -> False: Element is in b and a
    #   -> True:  Element is not in b, but only in a

    #Return only the elements
    return a[ mask ]

@jit(nopython=True, fastmath=True, parallel = False)
def evaluate_lagtimes(pos, lagtimes, N):

    #Storage arrays
    msd             = np.zeros(  lagtimes.shape[0],     dtype = np.float32)
    sd_per_particle = np.zeros( (lagtimes.shape[0], N), dtype = np.float32)

    nlags = len( lagtimes )

    #Evalulate lagtimes
    for i in prange(nlags):

        if i == 0: continue
        
        lag = lagtimes[i]

        dr = pos[:-lag] - pos[lag:]

        sqdist = np.square(dr).sum(axis=-1)

        msd[i]             = sqdist.mean()
        sd_per_particle[i] = np.sum(sqdist, axis = 0) / sqdist.shape[0]
    
    return msd, sd_per_particle


def MSD_fft_ax(pos):

    """
    Mean Square Displacement Calculation using Fourier Transforms

    The Mean Square Displacement Calculation can be significantly speed up using Fourier Transforms.
    The MSD equation,

    sum( ( r(k+m) - r(k) )^2 ) / (N - m)

    can be re-written as

    [ sum( r(k+m)^2 + r(k)^2 ) - 2 * sum( r(k) * r(k+m) ) ] / (N - m) = [ S1(m) - 2*S2(m) ] / (N - m)

    The part S2(m) denotes an autocorrelation function that can be solved with Fourier Transforms, while S1(m) can be solved recursively.
    

    
    Adapted from: https://stackoverflow.com/questions/69738376/how-to-optimize-mean-square-displacement-for-several-particles-in-two-dimensions/69767209#69767209

    """

    nTime=pos.shape[0]

    axis_time  = 0
    axis_coord = 2

    #Calculate the Power Spectral Density of the function
    FT  = np.fft.fft(pos, n=2*nTime, axis = axis_time)
    PSD = FT * FT.conjugate()

    #Autocorrelation -> Not normed yet
    S2  = np.sum( np.fft.ifft( PSD, axis = axis_time ).take(range(nTime), axis = axis_time).real, axis = axis_coord )

    #Calculate S1 via the Fast Correlation Algorithm
    D=np.square(pos).sum(axis=axis_coord) #Equal to r_x(k)^2 + r_y(k)^2

    shape_t     = (nTime, 1)
    shape_non_t = (1, pos.shape[1] )

    D=np.append(D, np.zeros( shape_non_t ), axis = axis_time) #[A, B, C, 0]

    Q = 2 * np.sum(D, axis = axis_time).reshape(shape_non_t)

    S1 = ( Q - 
          np.cumsum( 
                    np.insert( D[:-1, :], 0, 0, axis = axis_time) + 
                    np.flip(   D,               axis = axis_time),  #[0, A, B, C] + [0, C, B, A]
                    axis = axis_time )
          )[:-1, :]

    MSD = ( S1-2*S2 ) / ( nTime-np.arange(nTime).reshape(shape_t) )

    Dt_r = np.arange(1, nTime - 1)

    sd_per_particle = MSD.take(Dt_r, axis = axis_time)
    msd             = MSD.mean(1) 
    
    return msd, sd_per_particle



def distribute_domains_random_same_radius(number, r, size_x, size_y, pbc_dim, output):

    attempts     = 0
    max_attempts = 10000000

    r_buffer = r + 0.4

    number_placed = 0
    while number_placed < number:

        xy = np.random.rand(1, 2)

        xy[:, 0] *= size_x
        xy[:, 1] *= size_y

        if number_placed == 0: 
            placed_ = xy
            number_placed += 1
            continue

        d = placed_ - xy
        d = apply_pbc_vector(vec = d, pbc_dim = pbc_dim)

        d = np.linalg.norm(d, axis = 1)

        if np.all( d > 2 * r_buffer): 
            placed_ = np.vstack((placed_, xy))
            number_placed += 1

        attempts += 1

        if attempts == max_attempts: raise ValueError("Too many attempts to place domains")

    with open(f"{output}_random_midpoints.txt", "w") as f:

        f.write("Position of random domains\n")
        for i, placed_i in enumerate(placed_): f.write(f"c{i};{placed_i[0]};{placed_i[1]}\n")

    return placed_

def generate_grid(size_x, size_y, grid_spacing):
    
    x = np.linspace(0, size_x, int(np.round(size_x / grid_spacing) + 1))[:-1] + grid_spacing / 2
    y = np.linspace(0, size_y, int(np.round(size_y / grid_spacing) + 1))[:-1] + grid_spacing / 2

    nx = len(x)
    ny = len(y)

    xv, yv = np.meshgrid(x, y, indexing = "xy")

    xi, yi = np.meshgrid(range(nx), range(ny), indexing = "xy")

    grid     = np.column_stack((xv.ravel(), yv.ravel()))
    idx_grid = np.column_stack((xi.ravel(), yi.ravel()))

    full_grid= np.vstack((xv[np.newaxis, :, :], yv[np.newaxis, :, :])).T

    return grid, idx_grid, full_grid, nx, ny

@jit(nopython=True, fastmath = True)
def calc_vector_field( pos, grid, pbc_dim):
    
    #Apply PBC
    for i, size in zip(pbc_dim[0], pbc_dim[1]): pos[:, i] %= size
    
    vecs = pos[:, np.newaxis, :] - grid[np.newaxis, :, :]
    
    #Apply PBC
    for i, size in zip(pbc_dim[0], pbc_dim[1]):
        vecs[:, :, i] = np.where(vecs[:, :, i] >    size / 2, vecs[:, :, i] - size, vecs[:, :, i])
        vecs[:, :, i] = np.where(vecs[:, :, i] <= - size / 2, vecs[:, :, i] + size, vecs[:, :, i])

    vecs = np.sum( vecs**2, axis = 2)

    return vecs

def vector_field(pos, displacement, nx, ny, grid, idx_grid,  pbc_dim):

    vecs = calc_vector_field( pos = pos, grid = grid, pbc_dim = pbc_dim)

    nearest_grid_indices = vecs.argmin(-1)

    store_vector = np.zeros(( nx, ny, 2), dtype = np.float32)
    divid_vector = np.zeros(( nx, ny, 2), dtype = np.float32)

    k =0
    for ix, iy in zip(idx_grid[nearest_grid_indices, 0], idx_grid[nearest_grid_indices, 1]):

        store_vector[ ix, iy , :] += displacement[k]
        divid_vector[ ix, iy , :] += 1.

        k+=1

    return store_vector / divid_vector

#@jit(nopython=True)
def self_rdf(pos, pbc_dim, r_max, binwidth, exp_density):

    nFrames, N, _ = pos.shape

    print("Number of frames:", nFrames)
    print("Number of particles:", N)
    
    #RDF calculation
    bins          = np.linspace(0, r_max, int( np.round( r_max / binwidth + 1.0 ) ) )
    _, edges      = np.histogram( a = [], bins = bins )

    edges         = edges.astype(np.float32)
    
    shell_area    = np.pi * (edges[1:]**2 - edges[:-1]**2)
    binmids       = (edges[1:] + edges[:-1]) / 2

    rdf = np.zeros( (nFrames, len(binmids) ) )
    cdf = np.zeros( (nFrames, len(binmids) ) )

    for i in tqdm( range(nFrames) ):

        dist, vec = distance_matrix_NxN(pos = pos[i], N = N, pbc_dim = pbc_dim, offsets = np.zeros((N, N), dtype = np.float32))

        dist = dist.flatten()

        dist = dist[np.isfinite(dist)]

        hist, _ = np.histogram( a = dist, bins = bins )

        rdf[i]  = 2 * hist / shell_area / (N-1) / exp_density
        cdf[i]  = np.cumsum( 2 * hist / (N-1) )

    return binmids, rdf, cdf

def self_bulk_rdf(pos, pbc_dim, r_max, binwidth, exp_density, domain_coords, radii, rmin):

    nFrames, N, _ = pos.shape

    print("Number of frames:", nFrames)
    print("Number of particles:", N)
    
    #RDF calculation
    bins          = np.linspace(0, r_max, int( np.round( r_max / binwidth + 1.0 ) ) )
    _, edges      = np.histogram( a = [], bins = bins )

    edges         = edges.astype(np.float32)
    
    shell_area    = np.pi * (edges[1:]**2 - edges[:-1]**2)
    binmids       = (edges[1:] + edges[:-1]) / 2

    rdf = np.zeros( (nFrames, len(binmids) ) )
    cdf = np.zeros( (nFrames, len(binmids) ) )

    radii_sq = (radii + 3 * rmin)**2

    for i in tqdm( range(nFrames) ):

        dist2mids = apply_pbc_vector_3dim(vec = pos[i][:, None, :] - domain_coords[None, :, :], pbc_dim = pbc_dim)

        dist2mids = np.sum( dist2mids**2, axis = -1) 

        bulk_mask = np.all( dist2mids > radii_sq, axis = 1)
        N_eff     = bulk_mask.sum()

        dist, vec = distance_matrix_NxN(pos = pos[i][bulk_mask], N = N_eff, pbc_dim = pbc_dim, offsets = np.zeros((N_eff, N_eff), dtype = np.float32))

        dist = dist.flatten()

        dist = dist[np.isfinite(dist)]

        hist, _ = np.histogram( a = dist, bins = bins )

        rdf[i]  = 2 * hist / shell_area / (N_eff-1) / exp_density
        cdf[i]  = np.cumsum( 2 * hist / (N_eff-1) )

    return binmids, rdf, cdf
