import numpy as np

#Import custom modules
from . import utils

def clean_domains(init_pos, geometry_collection, N, lj_sig, pbc_dim, size_x, size_y):

    """
    The function should remove initials position of particles from domains (either hard boundaries or different diffusion coefficients).

    Parameters
    ----------

    init_pos := numpy.ndarray
        Array of initial positions that should be cleaned
    geometry_collection := dict
        Dictionary containing information about the domains geometries
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

    #Check if there are no geometry entries
    if not any( geometry_collection ) and lj_sig <= 0: return init_pos

    print('Found applied geometries!')
    print('Start resampling...')

    #Copy positions
    cleaned_init_pos = np.copy( init_pos )

    #Dummy array to start while loop
    in_bound_idx = np.array([np.inf])
    
    #Iterate as long as there are particles in domains
    while in_bound_idx.size > 0:
    
        #Init empty array to store particles indices in domains
        in_bound_idx = np.array([], dtype = np.int64)

        #Iterate over geometries
        for key, geometry in geometry_collection.items():

            if key == "Type": continue

            #Circular domains
            elif 'c' in key:

                #Get indices of particles in circular domains
                in_bound_idx_key = utils.check_circ_cond(pos     = cleaned_init_pos,
                                                         mid     = geometry[0].reshape(1, 2),
                                                         r       = (geometry[1] + 2**(1/6) * 0.6 * lj_sig)**2,
                                                         pbc_dim = pbc_dim)

            else: raise ValueError(f'Key {key} not known!') 
            
            #Append to larger storage array
            in_bound_idx = np.append( in_bound_idx, in_bound_idx_key )
        
        #--------------------------------------------------------------
        #Resample particles that are spatially to close
        if lj_sig > 0:

            dist_mat, vec_mat = utils.distance_matrix_NxN(pos = cleaned_init_pos, N = N, pbc_dim = pbc_dim)
            dist_mat          = dist_mat.min(axis = 1)

            close_idx = np.where( dist_mat < (lj_sig * 2**(1/6) * 0.6 )**2 )
        
            in_bound_idx = np.append( in_bound_idx, close_idx )
                
        print(f"There are {in_bound_idx.shape[0]} particles that need to be resampled! Please stay patienced!")
        
        #Resample only particle positions inside the boundaries
        cleaned_init_pos[ in_bound_idx ] = np.random.rand( in_bound_idx.shape[0], 2 )           

        #Scale the coordinates to the right size
        for i, size in enumerate([size_x, size_y]): cleaned_init_pos[ in_bound_idx , i] *= size
    
    print('Resampling finished! :-)')
    print('Good luck with your simulations!')

    return cleaned_init_pos
