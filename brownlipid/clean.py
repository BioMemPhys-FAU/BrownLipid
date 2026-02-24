import numpy as np

#Import custom modules
from . import utils

def clean_domains(init_pos, N, lj_sig, sigma_matrix, pbc_dim, hard_wall, soft_wall, size_x, size_y, offsets):

    """
    The function should remove initial positions of particles from domains (either hard boundaries or different diffusion coefficients).

    Parameters
    ----------

    init_pos := numpy.ndarray
        Array of initial positions that should be cleaned
    N := int
        Total number of particles in the system
    lj_sig := float
        Sigma value of the Lennard Jones parameter. Reflects minimum distance between two particles at equilibrium.
    pbc_dim := tuple
        First list contains index of dimensions along which PBC is applied. Second list contains length of box vector along which PBC is applied.
    size_x := float
        Length of box vector in x direction
    size_y := float
        Length of box vector in y direction

    Return
    ------

    cleaned_init_pos := numpy.ndarray
        Array of cleaned initial positions

    """

    #Check for sigma values
    if np.all(lj_sig <= 0): return init_pos

    print('Start resampling...')

    #Copy positions
    cleaned_init_pos = np.copy( init_pos )

    #Dummy array to start while loop
    in_bound_idx = np.array([np.inf])
    
    #Iterate as long as there are particles in domains
    while in_bound_idx.size > 0:
    
        #Init empty array for storing
        in_bound_idx = np.array([], dtype = np.int64)

        #Resample particles that are spatially too close
        if np.any(lj_sig > 0):

            dist_mat, vec_mat = utils.distance_matrix_NxN(pos = cleaned_init_pos, N = N, pbc_dim = pbc_dim, offsets = offsets)

            close_idx = np.where( dist_mat < (sigma_matrix * 2**(1/6) * 0.6 ) )
        
            in_bound_idx = np.append( in_bound_idx, close_idx[0])
        
        #--------------------------------------------------------------
        #Resample particles that are too close to a wall with a soft Lennard-Jones potential
        if np.any(lj_sig > 0) and soft_wall == True and len(pbc_dim[0]) < 2:

            for idx, size in zip( hard_wall[0], hard_wall[1]):

                close_idx = np.where( cleaned_init_pos[:, idx] < (np.diag(sigma_matrix) * 2**(1/6) ) )
                in_bound_idx = np.append( in_bound_idx, close_idx )
                
                close_idx = np.where( cleaned_init_pos[:, idx] > (size - (np.diag(sigma_matrix) * 2**(1/6) )) )
                in_bound_idx = np.append( in_bound_idx, close_idx )

        in_bound_idx = np.unique(in_bound_idx).astype(int)

        print(f"There are {in_bound_idx.shape[0]} particles that need to be resampled! Please stay patienced!")
        
        #Resample only particle positions inside the boundaries
        cleaned_init_pos[ in_bound_idx ] = np.random.rand( in_bound_idx.shape[0], 2 )           

        #Scale the coordinates to the right size
        for i, size in enumerate([size_x, size_y]): cleaned_init_pos[ in_bound_idx , i] *= size
    
    print('Resampling finished! :-)')
    print('Good luck with your simulations!')

    return cleaned_init_pos
