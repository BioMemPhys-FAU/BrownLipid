# ----PYTHON---- #
from typing import Union, Dict, Any

#Math
import numpy as np
from numba import jit

#Plots
import matplotlib.pyplot as plt

#Progress bar
from tqdm import tqdm 

#Own modules
from .base import base
from . import utils
from . import force
from . import reflection
from . import metropolis

class Universe(base):

    def evolve(self):

        """
        Main Function
        
        In this function, the positions of the particles are initialized and are then moved forward in time.
        For each step, the components of the displacement vector are randomly sampled from a gaussian distribution and are
        added to the actual position of the particles. Periodic boundary conditions, domains with different diffusion coefficients, and hard boundaries 
        are taken into account.

        There are two "universe" objects that are considered during the setup and the evolution of the system:

            - A "wrapped" universe (self.w_universe) in which positions are bounded by the limits of the box (self.size_x, and self.size_y).
            - An "unwrapped" universe (self.u_universe) in which positions are not bounded by the limits of the box. This can be used for further analysis (e.g., MSD)

        To prevent a slow down of the simulation due to an increase in the stored trajectory, two methods are applied:

            - Frames are stored only every self.nstxout steps in w_storage/u_storage
            - Arrays w_storage/u_storage are written to disk every self.nstchk steps.

        Therefore self.nstchk // self.nstxout frames are stored in a single checkpoint file.

        """

        #---------------------------------------------------------------------------------------------------------------------
        #Initialization Step

        #Populate the universe uniformly, but take hard boundaries into account
        self.w_universe = self.populate_universe_uniform()
        self.u_universe = np.copy( self.w_universe )       #Unwrapped, and wrapped universe are starting from the same positions
        
        #Number of frames in a checkpoint file
        nstchk_frames = self.nstchk // self.nstxout
        
        #Setup storage for positions -> Shape: (Number of frames in checkpoint file, Number of Particles, Number of dimensions)
        #The first checkpoint file has one frame more, because it contains also the init positions
        w_storage  = np.zeros( ( nstchk_frames + 1, self.N, 2 ), dtype = np.float32 )
        u_storage  = np.zeros( ( nstchk_frames + 1, self.N, 2 ), dtype = np.float32 )

        #Setup storage for time steps -> Shape: (Number of frames in checkpoint file)
        time_array = np.zeros(  nstchk_frames + 1, dtype = np.float32 )
        f_inside = np.zeros(  (nstchk_frames + 1, self.N), dtype = np.float32 )
        
        #Store initial frame
        w_storage[0] = self.w_universe
        u_storage[0] = self.u_universe
        
        #Store initial time
        time_array[0] = 0

        #Store initial fraction of particles in domains
        f_inside[0] = self.in_domains

        #---------------------------------------------------------------------------------------------------------------------
        #Main Iteration

        #Iterate over the requested number of steps.
        #It starts at 1, because time step 0 is the initial distribution of the points in the universe.
        #It ends at self.nsteps + 1 to ensure that the requested time is reached
        for i in tqdm(range(1, self.nsteps + 1) ):
            
            #-----------------------------------------------------------------------------------------------------------------
            #Evolve particle position

            self.frame = i

            #Move particles in time
            if not any(self.external_forces): self.forward_in_time()
            else: self.forward_in_time_external_forces()
            
            #Apply hard wall boundary conditions
            self.w_universe = self.apply_hard_wall(pos = self.w_universe)
            
            #Apply periodic boundary conditions
            self.w_universe = self.apply_pbc(pos = self.w_universe)
            
            #Apply hard boundaries
            self.hard_boundaries()
            
            #Check diffusion coefficients
            self.update_diffusion()

            #-----------------------------------------------------------------------------------------------------------------
            #Store particle positions

            #Unwrap self.w_universe and store the positions in self.u_universe
            for j, size in enumerate([self.size_x, self.size_y]):
                self.u_universe[:, j] = self.w_universe[:, j] - np.floor( (self.w_universe[:, j] - self.u_universe[:, j]) / size + 0.5 ) * size

            #Write current state of the universe into storage arrays
            if not i % self.nstxout: 

                #Calculate current time
                time = self.dt * i
                
                #--------------------------------------------------------------------------------------------------
                #Fill storage arrays

                #Current frame index in the current checkpoint file
                chk_frame_index = ( ( (i-1) // self.nstxout) % nstchk_frames ) + utils.heaviside(x = i, threshold = self.nstchk)
                
                w_storage[ chk_frame_index, :, : ] = self.w_universe
                u_storage[ chk_frame_index, :, : ] = self.u_universe
                
                time_array[ chk_frame_index ] = time

                f_inside[ chk_frame_index ] = self.in_domains 

                #If check pointing is requested then write current storage arrays to disk and renew them
                if not i % self.nstchk:
                
                    #Number of current check point
                    chk_number = str(i // self.nstchk)

                    #Write arrays to disk
                    np.save( arr = w_storage,  file = self.output +   f"_wrap.{chk_number.zfill(5)}" )
                    np.save( arr = u_storage,  file = self.output + f"_unwrap.{chk_number.zfill(5)}" )
                    
                    np.save( arr = time_array, file = self.output +   f"_time.{chk_number.zfill(5)}" )
                    
                    np.save( arr = f_inside, file = self.output +   f"_f_inside.{chk_number.zfill(5)}" )
        
                    #Setup storage for positions -> Shape: (Number of frames in checkpoint file, Number of Particles, Number of dimensions)
                    w_storage  = np.zeros( ( nstchk_frames, self.N, 2 ), dtype = np.float32 )
                    u_storage  = np.zeros( ( nstchk_frames, self.N, 2 ), dtype = np.float32 )

                    #Setup storage for time steps -> Shape: (Number of frames in checkpoint file)
                    time_array = np.zeros(  nstchk_frames, dtype = np.float32 )
                    
                    f_inside = np.zeros(  (nstchk_frames, self.N), dtype = np.float32 )
    
        print("Your simulation terminated successfully!")
        print("Have a nice day and thanks for the fish! :-)")


    #--------------------------------------------------------------------------------------------------------------
    # Function for initilization of the universe

    def populate_universe_uniform(self):

        """
        Generate an initial distribution of particles in the box.
        The function takes soft or hard domains into account and removes particle from this area, and resamples it.

        Returns
        -------

        init_pos := numpy.ndarray
            Array storing the initial positions of the universe.

        """

        #----------------------------------------------------------------------------------------------------------
        #Standard Workflow
        #Sample particle positions from random uniform distribution
        init_pos = np.random.rand( self.N, 2 )
        
        #Scale the coordinates according to the box lengths
        for i, size in enumerate([self.size_x, self.size_y]): init_pos[:, i] *= size
        
        #----------------------------------------------------------------------------------------------------------
        #Clean restricted areas
        #Take care of domains with hard boundaries

        if any(self.external_forces):
            if any( self.hard_boundaries_geometry ): init_pos = self.clean_domains_minimum_distance(init_pos = init_pos, geometry_collection = self.hard_boundaries_geometry)
            elif any( self.domain_geometry ): init_pos = self.clean_domains_minimum_distance(init_pos = init_pos, geometry_collection = self.hard_boundaries_geometry)
            else: init_pos = self.clean_domains_minimum_distance(init_pos = init_pos, geometry_collection = {})

        else:

             if any( self.hard_boundaries_geometry ): init_pos = self.clean_domains(init_pos = init_pos, geometry_collection = self.hard_boundaries_geometry)
             elif any( self.domain_geometry ): init_pos = self.clean_domains(init_pos = init_pos, geometry_collection = self.hard_boundaries_geometry)
             else: pass
        
        #----------------------------------------------------------------------------------------------------------
        #Make some tests if the particle coordinates are in the box
        assert init_pos[:, 0].max() <= self.size_x 
        assert init_pos[:, 1].max() <= self.size_y
        
        assert init_pos[:, 0].min() >= 0
        assert init_pos[:, 1].min() >= 0

        #----------------------------------------------------------------------------------------------------------
        #Save the initial frame to disk
        np.save(arr = init_pos, file = self.output + f"_initial_frame.npy")

        return init_pos

    def clean_domains(self, init_pos, geometry_collection):

        """
        The function should remove initials position of particles from domains (either hard boundaries or different diffusion coefficients).

        Parameters
        ----------

        init_pos := numpy.ndarray
            Array of initial positions that should be cleaned
        geometry_collection := dict
            Dictionary containing information about the domains geometries

        Return
        ------

        cleaned_init_pos := numpy.ndarray
            Array of cleaned initial positions

        """

        #Check if there are no geometry entries
        if not any( geometry_collection ): return init_pos

        print('Found applied geometries!')
        print('Start resampling...')

        cleaned_init_pos = np.copy( init_pos )

        check = True
        
        #Iterate as long as there are particles in domains
        while check:
        
            #Init empty array to store particles indices in domains
            in_bound_idx = np.array([], dtype = np.int64)

            #Iterate over geometries
            for key, geometry in geometry_collection.items():

                #Circular domains
                if 'c' in key:

                    #Get indices of particles in circular domains
                    in_bound_idx_key = utils.check_circ_cond(pos     = cleaned_init_pos,
                                                            mid     = geometry[0].reshape(1, 2),
                                                            r       = geometry[1],
                                                            pbc_dim = self.pbc_dim)

                #Rectangular domains
                elif 'p' in key:

                    #Get indices of particles in rectangular domains
                    in_bound_idx_key = utils.check_square_cond(pos = cleaned_init_pos,
                                                              Lx  = geometry[4],
                                                              Ly  = geometry[5],
                                                              mid = geometry[6].reshape(1, 2))  
                
                else: raise ValueError(f'Key {key} not known!') 
                
                #Append to larger storage array
                in_bound_idx = np.append( in_bound_idx, in_bound_idx_key )

            if not ( in_bound_idx.size > 0 ): 
                check = False
                continue

            print(f"There are {in_bound_idx.shape[0]} particles that need to be resampled! Please stay patienced!")
            
            #Resample only particle positions inside the boundaries
            cleaned_init_pos[ in_bound_idx ] = np.random.rand( in_bound_idx.shape[0], 2 )           

            #Scale the coordinates to the right size
            for i, size in enumerate([self.size_x, self.size_y]): cleaned_init_pos[ in_bound_idx , i] *= size
        
        print('Resampling finished! :-)')
        print('Good luck with your simulations!')

        return cleaned_init_pos
    
    def clean_domains_minimum_distance(self, init_pos, geometry_collection):

        """
        The function should remove initials position of particles from domains (either hard boundaries or different diffusion coefficients).

        Parameters
        ----------

        init_pos := numpy.ndarray
            Array of initial positions that should be cleaned
        geometry_collection := dict
            Dictionary containing information about the domains geometries

        Return
        ------

        cleaned_init_pos := numpy.ndarray
            Array of cleaned initial positions

        """

        print('Found applied geometries!')
        print('Start resampling...')

        cleaned_init_pos = np.copy( init_pos )

        check = True
        
        #Iterate as long as there are particles in domains
        while check:
        
            #Init empty array to store particles indices in domains
            in_bound_idx = np.array([], dtype = np.int64)

            #Iterate over geometries
            for key, geometry in geometry_collection.items():

                #Circular domains
                if 'c' in key:

                    #Get indices of particles in circular domains
                    in_bound_idx_key = utils.check_circ_cond(pos     = cleaned_init_pos,
                                                            mid     = geometry[0].reshape(1, 2),
                                                            r       = geometry[1],
                                                            pbc_dim = self.pbc_dim)

                #Rectangular domains
                elif 'p' in key:

                    #Get indices of particles in rectangular domains
                    in_bound_idx_key = utils.check_square_cond(pos = cleaned_init_pos,
                                                              Lx  = geometry[4],
                                                              Ly  = geometry[5],
                                                              mid = geometry[6].reshape(1, 2))  
                
                else: raise ValueError(f'Key {key} not known!') 
                
                #Append to larger storage array
                in_bound_idx = np.append( in_bound_idx, in_bound_idx_key )

            #--------------------------------------------------------------
            #Too close
            dist_mat, vec_mat = utils.distance_matrix_NxN(pos = cleaned_init_pos, N = self.N, pbc_dim = self.pbc_dim)
            dist_mat = dist_mat.min(axis = 1)

            close_idx = np.where( dist_mat < (self.lj_sig * 2**(1/6) * 0.7 )**2 )

            in_bound_idx = np.append( in_bound_idx, close_idx )

            if not ( in_bound_idx.size > 0 ): 
                check = False
                continue

            in_bound_idx = np.unique(in_bound_idx)

            print(f"There are {in_bound_idx.shape[0]} particles that need to be resampled! Please stay patienced!")
            
            #Resample only particle positions inside the boundaries
            cleaned_init_pos[ in_bound_idx ] = np.random.rand( in_bound_idx.shape[0], 2 )           

            #Scale the coordinates to the right size
            for i, size in enumerate([self.size_x, self.size_y]): cleaned_init_pos[ in_bound_idx , i] *= size
        
        print('Resampling finished! :-)')
        print('Good luck with your simulations!')

        return cleaned_init_pos
    
    
    #--------------------------------------------------------------------------------------------------------------
    # Function for the evolution of the system

    def forward_in_time(self):

        """
        Displace particle positions according to the Brown's Diffusion.
        For each particle and for every dimensions a random number from a standard normal distribution is drawn.
        The random number is then scaled by a factor taking a diffusion coefficient into account:

            sqrt( 2 * D * dt) (1)

        Equation 1 takes the diffusion coefficient (D) and the time step (dt) of the simulation.

        In this implementation D is an array (self.d_coeffs) that stores the diffusion coefficient of every particle.
        The diffusion coefficients can vary between the particles, e.g., if a particle is trapped in a domain.
        """

        #Calculate the factor for every particle.
        factor = np.sqrt( 2 * self.d_coeffs * self.dt ).reshape(-1, 1)

        #Calculate the displace vector
        self.displace = factor * np.random.randn( self.N, 2 )

        #Store previous positions
        self.w_universe_prev = np.copy( self.w_universe )

        #Move particles
        self.w_universe += self.displace

    def lennard_jones(self):

        if (self.frame % self.lj_nstlist) == 1: 
            rij, rij_sq, mask, self.pairlist = force.generate_pairlist(pos       = self.w_universe,
                                                                       pbc_dim   = self.pbc_dim,
                                                                       N         = self.N,
                                                                       lj_buffer = self.lj_buffer,
                                                                       lj_cutoff = self.lj_cutoff)
        else: 
            rij, rij_sq, mask = force.update_pairlist(pos       = self.w_universe,
                                                      pairlist  = self.pairlist,
                                                      pbc_dim   = self.pbc_dim,
                                                      lj_cutoff = self.lj_cutoff)

        assert rij_sq.min() >= 1E-12, f'Too small! {rij_sq.min()}'

        force_per_particle = force.calculate_force(rij             = rij,
                                                   rij_sq          = rij_sq,
                                                   N               = self.N,
                                                   lj_A12          = self.lj_A12,
                                                   lj_B6           = self.lj_B6,
                                                   masked_pairlist = self.pairlist[mask])



        return force_per_particle
    
    def forward_in_time_external_forces(self):

        """
        Displace particle positions according to the Brown's Diffusion.
        For each particle and for every dimensions a random number from a standard normal distribution is drawn.
        The random number is then scaled by a factor taking a diffusion coefficient into account:

            sqrt( 2 * D * dt) (1)

        Equation 1 takes the diffusion coefficient (D) and the time step (dt) of the simulation.

        In this implementation D is an array (self.d_coeffs) that stores the diffusion coefficient of every particle.
        The diffusion coefficients can vary between the particles, e.g., if a particle is trapped in a domain.
        """

        diff_dt = (self.d_coeffs * self.dt).reshape(-1, 1)

        #Calculate the factor for every particle.
        factor = np.sqrt( 2 * diff_dt )

        force = self.lennard_jones()

        #Calculate the displace vector
        self.displace = diff_dt * force / self.RT + factor * np.random.randn( self.N, 2 )

        #Store previous positions
        self.w_universe_prev = np.copy( self.w_universe )

        #Move particles
        self.w_universe += self.displace
    
    def apply_pbc(self, pos):

        """
        Periodic boundary conditions for POSITIONAL VECTORS a.k.a. POINTS.

        Particles leaving the box on one site, enter the box again from the opposite site.
        Since, a simple rectangular box shape is used as unit cell the modulo operator is applied here.

        """
        
        if not any( self.pbc_dim ): return pos
        
        assert pos.ndim == 2, 'Position vector has not a dimension of 2'

        #Apply periodic boundary conditions for every requested dimension
        for i, size in zip(self.pbc_dim[0], self.pbc_dim[1]): pos[:, i] %= size

        return pos
    
    def apply_hard_wall(self, pos):

        """
        Periodic boundary conditions for POSITIONAL VECTORS a.k.a. POINTS.

        Particles leaving the box on one site, enter the box again from the opposite site.
        Since, a simple rectangular box shape is used as unit cell the modulo operator is applied here.

        """
        
        if not any( self.hard_wall ): return pos

        assert pos.ndim == 2, 'Position vector has not a dimension of 2'

        #Apply periodic boundary conditions for every requested dimension
        for i, size in zip(self.hard_wall[0], self.hard_wall[1]): 
            
            pos[:, i] = np.where(pos[:, i] < 0   , -1 * pos[:, i]       , pos[:, i])
            pos[:, i] = np.where(pos[:, i] > size,  2 * size - pos[:, i], pos[:, i])

        return pos
    
    #--------------------------------------------------------------------------------------------------------------
    # Function for different geometric tasks
    
    #--------------------------------------------------------------------------------------------------------------
    # Function for different geometric tasks

    def update_diffusion(self):
        
        if not any(self.domain_geometry): pass

        else:

            self.d_coeffs[:] = self.base_d_coeff

            for key, geometry in self.domain_geometry.items():

                #Circular domains
                if 'c' in key:
                    
                    #Calculate real distances
                    #--------------------------------------------------------------------------------
                    mid = self.domain_geometry[key][0].reshape(1,2)
                    r   = self.domain_geometry[key][1]
                    #--------------------------------------------------------------------------------

                    index = utils.check_circ_cond(pos     = self.w_universe,
                                                 mid     = mid,
                                                 r       = r,
                                                 pbc_dim = self.pbc_dim)
                    
                    if not index.size > 0: continue

                    self.d_coeffs[ index ] = self.domain_geometry[key][2]
                
                #Square hard boundary
                elif 'p' in key:
                    
                    #Calculate real distances
                    #--------------------------------------------------------------------------------
                    edges = self.domain_geometry[key][0:4]
                    Lx    = self.domain_geometry[key][4]
                    Ly    = self.domain_geometry[key][5] 
                    mid   = self.domain_geometry[key][6].reshape(1, 2)
                    #--------------------------------------------------------------------------------
                    
                    index = utils.check_square_cond(pos = self.w_universe,
                                                   Lx = Lx,
                                                   Ly = Ly,
                                                   mid = mid)


                    if not index.size > 0: continue
                    
                    self.d_coeffs[ index ] = self.domain_geometry[key][7]

                else:
                    raise ValueError("Currently I cannot handle the provided geometry.")

    #----------------------------------------------------------------------------------------------------------------------------------------
    
    def double_metropolis_scheme(self, index, prev_index ):

        """
        Function calls Metropolis Algorithm to decide:

        A) Particle entering a domain
        B) Particle leaving a domain

        The implemention distinguish between four cases for each particle:

                               __ A.1 Particle was already in domain in the step before -> Particle remains in domain
                              |
        A) Particle in domain 
                              |__ A.2 Particle was not in domain in the step before -> Particle wants to enter domain -> Metropolis

                                  __ B.1 Particle was already in domain in the step before -> Particle wants to leave domain -> Metropolis
                                 |
        B) Particle not in domain
                                 |__ B.2 Particle was not in domain in the step before -> Particle stays outside

        The array self.in_domains contains information about the assignment of particles to domains in the previous step, but not about
        the current. self.in_domains is updated during this function.

        Parameters
        ----------
        index := numpy.ndarray
            Index of particles in the domain in the current step
        prev_index := numpy.ndarray
            Index of particles in the domain in the previous step

        Returns
        -------
        index_after_metropolis := numpy.ndarray
            Index of particles which Metropolis step got rejected. Array is further used for reflection.

        """
        
        #Case A
        A = self.in_domains[ index ]
        #Case A.1 ->  TRUE: Particle was already in this domain and remains there -> Do not reflect
        #Case A.2 -> FALSE: Particle was not in a domain and wants to enter this domain now -> METROPOLIS

        #Case B
        not_index = np.setdiff1d(ar1 = prev_index, ar2 = index) #Return the unique values in ar1 that are not in ar2.
        B = self.in_domains[ not_index ]
        #Case B.1 -> TRUE:  Particle wants to leave domain -> METROPOLIS
        #Case B.2 -> FALSE: Particle stays outside domain -> Do not reflect -> That should only happen if the point in a previous iteration was allowed to leave the domain.
        

        #Sort particles for which Metropolis must be called
        entering_index = index[ ~A ] #Case A.2
        leaving_index  = not_index[ B ] #Case B.1

        #Some tests
        assert np.all( B ) == True, 'That should not happen with circles! - 1'
        assert np.all( leaving_index == not_index), 'That should not happen with circles! - 2'
        
        #Init empty list to collect indices of particles with rejected Metropolis Step
        index_after_metropolis = []

        #-----------------------------------------------------------------------------------------------------------------------
        #OUTSIDE -> INSIDE
        
        #Decide for each "entering-applicant" particle independently if it is allowed to enter the domain
        #Iterate over A.2
        for p_index in entering_index:

            if self.target_fraction != None:
                x_old = self.in_domains.sum()     
                x_new = self.in_domains.sum() + 1
                deltaE = metropolis.get_delta_E(x_old = x_old, x_new = x_new, N = self.N, f_true = self.target_fraction, forceconstant = self.fconstant)

            elif self.barrier != None:
                deltaE = self.barrier

            else: raise ValueError('Could not handle request!')
            
            #Entering is accepted
            if metropolis.metropolis_decision(deltaE = deltaE, RT = self.RT): self.in_domains[ p_index ] = True
            #Entering is rejected
            else: index_after_metropolis.append( p_index ) #Particle will get reflected
        

        #INSIDE -> OUTSIDE

        #Decide for each particle indepently if it is allowed to enter the domain
        #Iterate over B.1
        for p_index in leaving_index:

            if self.target_fraction != None:
                x_old = self.in_domains.sum()     
                x_new = self.in_domains.sum() - 1
                deltaE = metropolis.get_delta_E(x_old = x_old, x_new = x_new, N = self.N, f_true = self.target_fraction, forceconstant = self.fconstant)

            elif self.barrier != None:
                deltaE = self.barrier

            else: raise ValueError('Could not handle request!')
            
            #Leaving is accepted
            if metropolis.metropolis_decision(deltaE = deltaE, RT = self.RT): self.in_domains[ p_index ] = False
            #Leaving is rejected
            else: index_after_metropolis.append( p_index )

        return np.array(index_after_metropolis)


    #-----------------------------------------------------------------------------------------------------------------------------
    #Boundaries
    
    def hard_boundaries(self):

        """
        Function to handle domains with hard boundaries.
        For particles crossing such a boundary, either to leave or to enter a domain, a Metropolis step is performed.
        If accepted, the particle is allowed to cross the boundary.
        If rejected, the particle is reflected depending on the geometry of the domain. 


        """
        
        #If called but no geometry for a domain is stored
        if not any(self.hard_boundaries_geometry): pass

        else:

            #---------------------------------------------------------------------------------------
            #Iterate over stored geometries
            for key, geometry in self.hard_boundaries_geometry.items():

                #Circular hard boundary
                if 'c' in key:
                    
                    #--------------------------------------------------------------------------------
                    #Calculate real distances
                    mid = self.hard_boundaries_geometry[key][0].reshape(1,2)
                    r   = self.hard_boundaries_geometry[key][1]
                    
                    index        = utils.check_circ_cond(pos = self.w_universe     , mid = mid, r = r, pbc_dim = self.pbc_dim)
                    prev_index   = utils.check_circ_cond(pos = self.w_universe_prev, mid = mid, r = r, pbc_dim = self.pbc_dim)
                    
                    #--------------------------------------------------------------------------------
                    #Perform Metropolis step if required
                    if any(self.metropolis): index = self.double_metropolis_scheme( index = index, prev_index = prev_index)

                    #--------------------------------------------------------------------------------
                    #Check if particles are reflected
                    if not index.size > 0: continue
                    
                    #--------------------------------------------------------------------------------
                    #Calculate reflection
                    #These are the indices of the particles that are reflected on the inside of the domain
                    index_in_domains = index[ self.in_domains[index] ]

                    #Calculate normals and intersection with the circular boundary
                    norm, intersection = reflection.get_normals_circle(points      = self.w_universe[index],
                                                                 prev_points = self.w_universe_prev[index],
                                                                 d           = self.displace[index],
                                                                 mid         = mid,
                                                                 r           = r,
                                                                 pbc_dim     = self.pbc_dim
                                                                 )
                    
                    new_pos, d_new = reflection.calc_reflection(intersection = intersection, p_in = self.w_universe[index], n = norm, pbc_dim = self.pbc_dim )
                    
                    if not np.all( index_in_domains == index[ utils.check_circ_cond(pos = new_pos, mid = mid, r = r, pbc_dim = self.pbc_dim) ] ):
                        print('Error points are not in circle, but are expected to be in circle!')

                        print(index)
                        print(index[ utils.check_circ_cond(pos = new_pos, mid = mid, r = r, pbc_dim = self.pbc_dim) ])

                        np.save(file = self.output + "_debug_p.npy"           , arr = new_pos)
                        np.save(file = self.output + "_debug_p_older.npy"     , arr = self.w_universe_prev[index])
                        np.save(file = self.output + "_debug_p_old.npy"       , arr = self.w_universe[index])
                        np.save(file = self.output + "_debug_intersection.npy", arr = intersection)
                        raise ValueError('')
                    
                    #Update positions
                    self.w_universe[index] = new_pos
                    
                #TODO: Check polygon reflection
                #Square hard boundary
                elif 'p' in key:
                    
                    #Calculate real distances
                    #--------------------------------------------------------------------------------
                    edges = self.hard_boundaries_geometry[key][0:4]
                    Lx    = self.hard_boundaries_geometry[key][4]
                    Ly    = self.hard_boundaries_geometry[key][5] 
                    mid   = self.hard_boundaries_geometry[key][6].reshape(1, 2)
                    #--------------------------------------------------------------------------------
                    
                    condition = True
                    counter = 0
                    while condition:

                        index      = utils.check_square_cond(pos = self.w_universe     , Lx = Lx, Ly = Ly, mid = mid)
                        prev_index = utils.check_square_cond(pos = self.w_universe_prev, Lx = Lx, Ly = Ly, mid = mid)
                        
                        if any(self.metropolis): index = self.double_metropolis_scheme( index, prev_index )
                        
                        if not index.size > 0: 
                            condition = False
                            continue

                        #--------------------------------------------------------------------------------

                        norm, intersection = self.get_normals_polygon(points   = self.w_universe[ index ],
                                                                      displace = self.displace[ index ],
                                                                      edges    = edges,
                                                                      Lx       = Lx,
                                                                      Ly       = Ly)
                
                        new_pos, d_new = self.calc_reflection(intersection = intersection, p_in = self.w_universe[index], n = norm , pbc_dim = self.pbc_dim)
                
                        self.w_universe_prev[index] = intersection
                        self.displace[index]        = d_new
                        self.w_universe[index]      = new_pos
                        
                        counter += 1
                        
                        assert counter < 2, 'Weird'
                        
                
                else:
                    raise ValueError("Currently I cannot handle the provided geometry.")
    
    

    #---------------------------------------------------------------------------------------------------------------------
    #Normal calculation

    def get_point_between_points(self, p, d, e1, e2, Lx, Ly):

        c = e1 - p
        c = utils.apply_pbc_vector(c)
        
        vert = (e2 - e1)
        vert = utils.apply_pbc_vector(vert.reshape(1, -1) )[0]
        
        if vert.sum() == 0: vert = (e2 - e1)

        t = utils.perpDot(c , vert) / utils.perpDot(d, vert)

        intersection = p + t.reshape(-1, 1) * d

        #----------------------------------------------------------
        true_intersection = np.ones( p.shape[0] , dtype = bool)
        #Check if point is on vector

        e1_inter = intersection - e1
        e1_inter = utils.apply_pbc_vector(e1_inter)
        
        e2_inter = intersection - e2
        e2_inter = utils.apply_pbc_vector(e2_inter)

        e1_inter2 = np.sum((e1_inter)**2, axis = 1)
        e2_inter2 = np.sum((e2_inter)**2, axis = 1)
        e1_e22    = np.sum(vert**2)

        check = e1_inter2 + e2_inter2 + 2 * np.sqrt(e1_inter2 * e2_inter2)

        cond = (check - e1_e22) >= 1E-8
        
        true_intersection[ cond ] = False
        
        #---------------------------------------------------------
        #Check if point is on vector

        p_inter = intersection - p
        p_inter = utils.apply_pbc_vector(p_inter)

        prev_pos = p-d

        prev_pos = self.apply_pbc(pos = prev_pos.reshape(-1, 2))
        
        prev_inter = intersection - prev_pos
        prev_inter = utils.apply_pbc_vector(prev_inter)

        p_inter2    = np.sum(   (p_inter)**2, axis = 1)
        prev_inter2 = np.sum((prev_inter)**2, axis = 1)
        prev_p2      = np.sum(d**2, axis = 1)

        check = p_inter2 + prev_inter2 + 2 * np.sqrt(p_inter2 * prev_inter2)

        cond = np.abs(check - prev_p2) >= 1E-8
        
        true_intersection[ cond ] = False
        #---------------------------------------------------------
        
        cond = np.abs( np.sqrt(prev_inter2) ) <= 1E-8
        true_intersection[ cond ] = False
        
        intersection[~true_intersection] = np.array([1E-8, 1E-8])


        return intersection, true_intersection, vert

    def get_normals_polygon(self, points, displace, edges, Lx, Ly):

        """
        Get the normal vector of a two dimensional two dimension polygon for a point

        points := numpy.ndarray
            coordinates of reflected points
        edges := list
            list of polygon edges
        
        """

        edges = edges + [edges[0]]
        
        number_of_vertices = len(edges) - 1

        final_norm         = np.zeros( (points.shape[0], 2), dtype = np.float32)
        final_intersection = np.zeros( (points.shape[0], 2), dtype = np.float32)

        for i in range( len(edges) - 2 + 1 ):

            polygon_point_a = edges[i]
            polygon_point_b = edges[i+1]
    
            intersection, true_intersection, a_to_b = self.get_point_between_points(p = points,
                                                                                    d = displace,
                                                                                   e1 = polygon_point_a,
                                                                                   e2 = polygon_point_b,
                                                                                   Lx = Lx,
                                                                                   Ly = Ly)

            if   a_to_b.sum() > 0.0: norm = polygon_point_a - intersection

            elif a_to_b.sum() < 0.0: norm = polygon_point_b - intersection

            else: raise ValueError(f"Fuck. Value of vertices: {a_to_b}. A: {polygon_point_a} B: {polygon_point_b}")
            
            norm = utils.apply_pbc_vector(norm)
            norm = np.flip(norm, axis = 1)
            norm[:, 0] *= -1

            norm /= np.linalg.norm(norm, axis = 1).reshape(-1, 1)

            norm[~true_intersection] = np.array([0.0, 0.0])


            final_norm += norm
            final_intersection += intersection

        len_norm = np.linalg.norm(final_norm, axis = 1)

        if not ( np.all( np.abs( len_norm - 1.) < 1E-8) == True and np.all( final_intersection[:, 0] < self.size_x ) == True and np.all( final_intersection[:, 1] < self.size_y ) == True):

            print('Norm')
            print(final_norm)
            print('Intersection')
            print(final_intersection)
            print('Positions')
            print(points)
            print('Prev Points')
            print(points-displace)
            
            print( 'Check' )
            print( len_norm[ np.abs( len_norm - 1.) >= 1E-8 ] )

            np.save(arr = points, file = 'debug_points.npy')
            np.save(arr = displace, file = 'debug_displace.npy')
            np.save(arr = final_intersection, file = 'debug_intersection.npy')

            raise ValueError("Normals are not normalized! Saved actual states as debug_points.npy and debug_displace.npy!")

        return final_norm, final_intersection
    
                    

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Analysis part

    def load_data_unwrap(self):

        try:
            if self.u_storage.shape[0] == self.nsteps // self.nstxout: return 0
        except: pass
        
        self.u_storage  = np.zeros( (0, self.N, 2), dtype = np.float32 )
        self.time_array = np.zeros( (0)                   , dtype = np.float32 )

        for chk_number in range(1, self.nsteps // self.nstchk + 1 ):

            chk_number = str(chk_number)

            self.u_storage  = np.vstack( (self.u_storage, np.load(self.output + f"_unwrap.{chk_number.zfill(5)}.npy") ))

            self.time_array = np.append( self.time_array, np.load(self.output +   f"_time.{chk_number.zfill(5)}.npy")  ) 

        #Validation
        assert self.u_storage.shape[0] == (self.nsteps // self.nstxout) + 1, "Number of frames is not correct!"

        assert np.allclose( np.diff( self.time_array ), self.dt * self.nstxout ), "Time step distance is not as expected!"
        assert np.allclose( self.dt * self.nstxout, np.diff( self.time_array ) ), "Time step distance is not as expected!"

        return 1
    
    def load_data_wrap(self):

        try:
            if self.w_storage.shape[0] == self.nsteps // self.nstxout: return 0
        except: pass
        
        self.w_storage  = np.zeros( (0, self.N, 2 ), dtype = np.float32 )
        self.time_array = np.zeros( (0)                   , dtype = np.float32 )

        for chk_number in range(1, self.nsteps // self.nstchk + 1 ):

            chk_number = str(chk_number)

            try: 

                self.w_storage  = np.vstack( (self.w_storage, np.load(self.output + f"_wrap.{chk_number.zfill(5)}.npy") ))
                self.time_array = np.append( self.time_array, np.load(self.output +   f"_time.{chk_number.zfill(5)}.npy")  ) 

            except FileNotFoundError as e:
                pass

        #Validation
        assert self.w_storage.shape[0] == (self.nsteps // self.nstxout) + 1, "Number of frames is not correct!"

        assert np.allclose( np.diff( self.time_array ), self.dt * self.nstxout ), "Time step distance is not as expected!"
        assert np.allclose( self.dt * self.nstxout, np.diff( self.time_array ) ), "Time step distance is not as expected!"

        return 1


    def mean_square_displacement(self, skip, begin = 0, stop = None):

        """
        Mean Square Displacement

        Calculate the Mean Square Displacement of the particles for different lag times.

        Parameters
        ----------

        skip := float
            Skip lag times to decrease calculation time (ns)
        begin := float
            Start time for analysis (ns)
        stop := float
            Stop time for analysis (ns)
        """
        
        if stop == None: stop = self.nsteps * self.dt
 
        assert stop <= self.nsteps * self.dt , f'Error. There are only {self.nsteps * self.dt} ns simulation time!' 
        assert begin <= self.nsteps * self.dt, f'Error. There are only {self.nsteps * self.dt} ns simulation time!'
        
        #Convert time to frames
        begin = int( np.round( begin / self.dt / self.nstxout ) )
        stop  = int( np.round( stop  / self.dt / self.nstxout ) )
        skip  = int( np.round( skip  / self.dt ) )

        #assert (self.nstxout % skip) == 0, 'Skip must be a multiple of nstxout'
        
        #Number of steps for analysis
        nsteps_analysis = stop - begin

        print(f"Calculating MSD...")
        print(f"Start: Frame {begin}")
        print(f"Stop : Frame {stop}")
        print(f"Skip : Frame {skip}")
        print(f"Frames: {nsteps_analysis}")
        print("")


        #Load data
        self.load_data_unwrap()

        #Lag times at which MSD is evaluated
        lagtimes = np.arange(0, nsteps_analysis, skip)

        u_storage_analysis = np.copy( self.u_storage[begin:stop, :, :] )

        assert nsteps_analysis == u_storage_analysis.shape[0], 'Not correct number of frames'

        msd, sd_per_particle = self.evaluate_lagtimes(pos = u_storage_analysis, lagtimes = lagtimes, N = self.N)
        

        #Convert lagtimes array to physical time
        tau = np.float32(lagtimes)
        tau = tau * self.dt * self.nstxout

        return tau, msd, sd_per_particle

    @staticmethod
    @jit(nopython=True, parallel=True)
    def evaluate_lagtimes(pos, lagtimes, N):
        
        #Storage arrays
        msd             = np.zeros(  lagtimes.shape[0],     dtype = np.float32)
        sd_per_particle = np.zeros( (lagtimes.shape[0], N), dtype = np.float32)

        #Evalulate lagtimes
        for i, lag in enumerate(lagtimes):

            if i == 0: continue

            dr = pos[:-lag, :, :] - pos[lag:, :, :]

            sqdist = np.square(dr).sum(axis=-1)

            msd[i]             = sqdist.mean()
            sd_per_particle[i] = np.sum(sqdist, axis = 0) / sqdist.shape[0]
        
        return msd, sd_per_particle
    
    def mean_square_displacement_1d(self, direction, skip, begin = 0, stop = None):

        """
        Mean Square Displacement in one dimension

        Calculate the Mean Square Displacement of the particles for different lag times.

        Parameters
        ----------

        direction := str
            Direction (x,y) in which the MSD is calculated
        skip := float
            Skip lag times to decrease calculation time (ns)
        begin := float
            Start time for analysis (ns)
        stop := float
            Stop time for analysis (ns)
        """

        if direction == 'x': direction = 0
        elif direction == 'y': direction = 1
        else: raise ValueError('Could not handle request! Use x or y!')
        
        if stop == None: stop = self.nsteps * self.dt
 
        assert stop <= self.nsteps * self.dt , f'Error. There are only {self.nsteps * self.dt} ns simulation time!' 
        assert begin <= self.nsteps * self.dt, f'Error. There are only {self.nsteps * self.dt} ns simulation time!'
        
        #Convert time to frames
        begin = int( np.round( begin / self.dt / self.nstxout ) )
        stop  = int( np.round( stop  / self.dt / self.nstxout ) )
        skip  = int( np.round( skip  / self.dt ) )

        assert (self.nstxout % skip) == 0, 'Skip must be a multiple of nstxout'
        
        #Number of steps for analysis
        nsteps_analysis = stop - begin

        print(f"Calculating MSD...")
        print(f"Start: Frame {begin}")
        print(f"Stop : Frame {stop}")
        print(f"Skip : Frame {skip}")
        print(f"Frames: {nsteps_analysis}")
        print("")


        #Load data
        self.load_data_unwrap()

        #Lag times at which MSD is evaluated
        lagtimes = np.arange(0, nsteps_analysis, skip)

        #Storage arrays
        msd             = np.zeros(  lagtimes.shape[0],          dtype = np.float32)
        sd_per_particle = np.zeros( (lagtimes.shape[0], self.N), dtype = np.float32)

        u_storage_analysis = np.copy( self.u_storage[begin:stop, :, direction] )

        assert nsteps_analysis == u_storage_analysis.shape[0], 'Not correct number of frames'

        #Evalulate lagtimes
        for i, lag in enumerate(lagtimes):

            if i == 0: continue

            dr = u_storage_analysis[:-lag, :] - u_storage_analysis[lag:, :]

            sqdist = np.square(dr)

            msd[i]             = sqdist.mean()
            sd_per_particle[i] = sqdist.mean(axis = 0)

        #Convert lagtimes array to physical time
        tau = np.float32(lagtimes)
        tau = tau * self.dt * self.nstxout

        return tau, msd, sd_per_particle
    
    def mean_square_displacement_distr(self, tau, begin = 0, stop = None):
        
        if stop == None: stop = self.nsteps * self.dt
 
        assert stop <= self.nsteps * self.dt , f'Error. There are only {self.nsteps * self.dt} ns simulation time!' 
        assert begin <= self.nsteps * self.dt, f'Error. There are only {self.nsteps * self.dt} ns simulation time!'
        assert (stop - begin) >= tau, 'Error. Time lag is larger than time interval!'

        #Convert time to frames
        begin = int( np.round( begin / self.dt / self.nstxout ) )
        stop  = int( np.round( stop  / self.dt / self.nstxout ) )
       
        #Load data
        self.load_data_unwrap()
        
        u_storage_analysis = self.u_storage[begin:stop]

        lag = int( np.round(tau / self.dt / self.nstxout) )

        dr = self.u_storage[:-lag, :, :] - self.u_storage[lag:, :, :]
        sqdist = np.square(dr).sum(axis=-1)

        sd_per_particle = sqdist.mean(axis = 0)

        print(f'Analysis from {begin * self.dt * self.nstxout / 1000 / 1000} to {stop * self.dt * self.nstxout / 1000 / 1000}')
        print(f'MSD Distribution at {lag * self.dt * self.nstxout / 1000 / 1000} ms') 

        return sd_per_particle
    
    @staticmethod
    def mean_square_displacement_fit(tau, msd, dim = 2):
        
        #tau -> ns
        #msd -> nm2
        DiffCoeff, Intercept = np.polyfit(x = tau, y = msd, deg = 1)

        DiffCoeff /= (2 * dim)

        #DiffCoeff -> nm2/ns -> um2/ms
        #Intercept -> nm2

        return DiffCoeff, Intercept

    @staticmethod
    def plot_log_histogram_mean_square_displacement_distr(sd_per_particle, label, lo_limit = 1E-4, up_limit = 1.0, nbins = 51, color = 'red'):

        """
        Plot histogram with log-space bins

        Parameters
        ----------

        sd_per_particle := numpy.ndarray
            Squared-Displacement per particle at a single time-lag tau (expected unit: square-micrometer)
        label           := str
            Label for legend
        lo_limit        := float
            Lower limit for log-spaced bins (opt.)
        up_limit        := float
            Upper limit for log-spaced bins (opt.)
        nbins           := int
            Number of log-spaced bins (opt.)
        color           := str
            Matplotlib color
            
        """

        a = plt.hist(sd_per_particle, 
                     density=True, 
                     bins = np.logspace(np.log10(lo_limit),np.log10(up_limit), nbins),
                     histtype = 'step',
                     color = color,
                     label = label
                    )

        #Scale
        plt.xscale('log')

        #Label
        plt.ylabel('Number of Trajectories')
        plt.xlabel(r'MSD / $\mu$m$^2$')

        


