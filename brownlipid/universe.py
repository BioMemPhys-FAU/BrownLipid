# ----PYTHON---- #
from typing import Union, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm 

from .base import base

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
                chk_frame_index = ( ( (i-1) // self.nstxout) % nstchk_frames ) + self.heaviside(x = i, threshold = self.nstchk)
                
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

    @staticmethod
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
                    in_bound_idx_key = self.check_circ_cond(pos = cleaned_init_pos,
                                                            mid = geometry[0].reshape(1, 2),
                                                            r   = geometry[1] )

                #Rectangular domains
                elif 'p' in key:

                    #Get indices of particles in rectangular domains
                    in_bound_idx_key = self.check_square_cond(pos = cleaned_init_pos,
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
                    in_bound_idx_key = self.check_circ_cond(pos = cleaned_init_pos,
                                                            mid = geometry[0].reshape(1, 2),
                                                            r   = geometry[1] )

                #Rectangular domains
                elif 'p' in key:

                    #Get indices of particles in rectangular domains
                    in_bound_idx_key = self.check_square_cond(pos = cleaned_init_pos,
                                                              Lx  = geometry[4],
                                                              Ly  = geometry[5],
                                                              mid = geometry[6].reshape(1, 2))  
                
                else: raise ValueError(f'Key {key} not known!') 
                
                #Append to larger storage array
                in_bound_idx = np.append( in_bound_idx, in_bound_idx_key )

            #--------------------------------------------------------------
            #Too close
            dist_mat, vec_mat = self.distance_matrix_NxN(pos = cleaned_init_pos)
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

    def distance_matrix_NxN(self, pos):

        #Improvements taken from:
        #https://github.com/Allen-Tildesley/examples/blob/master/python_examples/md_lj_module.py

        dist_mat = np.ones((self.N, self.N)   , dtype = np.float32) * np.inf
        vec_mat  = np.ones((self.N, self.N, 2), dtype = np.float32) * np.inf

        #Fill only upper triangle
        for i in range(self.N - 1):

            rij    = pos[i, :] - pos[(i+1):, :] 
            rij    = self.apply_pbc_vector( rij )

            rij_sq = np.sum(rij**2,axis=1)

            dist_mat[i, (i+1):]    = rij_sq
            vec_mat[ i, (i+1):, :] = rij

        #dist_mat[range(self.N), range(self.N) ] = np.inf

        return dist_mat, vec_mat

    def generate_pairlist(self):

        #------------------------------------------------
        #Setup pair list

        dist_mat, vec_mat = self.distance_matrix_NxN(pos = self.w_universe)

        pairlist = np.vstack( np.where( dist_mat <= self.lj_buffer ) ).T

        #First column always smaller than second column
        pairlist = np.sort(pairlist, axis = 1)

        self.pairlist = np.unique(pairlist, axis = 0)

        #------------------------------------------------
        #Buffer

        rij    =  vec_mat[ self.pairlist[:, 0], self.pairlist[:, 1] ]
        rij_sq = dist_mat[ self.pairlist[:, 0], self.pairlist[:, 1] ]

        assert np.all(np.isfinite(rij))     , 'Inf or nan in vector matrix!'
        assert np.all(np.isfinite(rij_sq)), 'Inf or nan in distance matrix!'

        #------------------------------------------------
        #Cutoff
        mask = (rij_sq <= self.lj_cutoff)

        rij    = rij[mask]
        rij_sq = rij_sq[mask]
        
        return rij, rij_sq, mask

    def update_pairlist(self):

        #Calculate distances
        #Vector pointing from 1 to 0. Force is acting on self.pairlist[:, 0]
        rij    = self.w_universe[ self.pairlist[:, 0] ] - self.w_universe[ self.pairlist[:, 1]]
        rij    = self.apply_pbc_vector( rij )

        rij_sq = np.sum(rij**2, axis = 1)
        
        mask = (rij_sq <= self.lj_cutoff)

        vec_r      = rij[mask]
        vec_r_norm = rij_sq[mask]

        return vec_r, vec_r_norm, mask

    def lennard_jones(self):

        if (self.frame % self.lj_nstlist) == 1: rij, rij_sq, mask = self.generate_pairlist()
        else: rij, rij_sq, mask = self.update_pairlist()

        assert rij_sq.min() >= 1E-12, f'Too small! {rij_sq.min()}'

        inv_rij_sq = 1.0 / rij_sq

        #Calculate powers
        sr6  = inv_rij_sq ** 3
        sr12 = sr6 ** 2

        force = (self.lj_A12 * sr12 - self.lj_B6 * sr6 ) * inv_rij_sq 
        force = force.reshape(-1, 1) * rij

        force_per_particle = np.zeros( (self.N, 2), dtype = np.float32 )

        k = 0
        for pair in self.pairlist[mask]:

            i, j = pair[0], pair[1]

            force_per_particle[i] += force[k]
            force_per_particle[j] -= force[k]

            k += 1

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
    
    def apply_pbc_vector(self, vec):

        """
        Periodic boundary conditions for DIRECTIONAL VECTORS a.k.a. VECTORS.

        If a component of a vector is larger than half the box size (positive and negative), subtract one box size.

        """
        
        if not any( self.pbc_dim ): return vec

        assert vec.ndim == 2, 'Vector has not a dimension of 2'
        
        #Apply PBC
        for i, size in zip(self.pbc_dim[0], self.pbc_dim[1]):
            vec[:, i] = np.where(vec[:, i] >    size / 2, vec[:, i] - size, vec[:, i])
            vec[:, i] = np.where(vec[:, i] <= - size / 2, vec[:, i] + size, vec[:, i])

        return vec

    def check_circ_cond(self, pos, mid, r):

        """
        Check if a particle is within a circle.

        Parameters
        ----------
        

        """

        assert mid.ndim == 2, 'Middle point is not two-dimensional'

        circ_coor = pos - mid
        circ_coor = self.apply_pbc_vector( circ_coor )

        circ_dist = np.linalg.norm(circ_coor, axis = 1)

        return np.where(circ_dist < r)[0]
        

    def check_square_cond(self, pos, Lx, Ly, mid): 
        
        assert mid.ndim == 2, 'Middle point is not two-dimensional'

        squa_coor = (pos - mid)
        squa_coor = self.apply_pbc_vector( squa_coor )

        cond_x = np.logical_and( (-Lx/2 <= squa_coor[:, 0]), (squa_coor[:, 0] <= Lx/2) )
        cond_y = np.logical_and( (-Ly/2 <= squa_coor[:, 1]), (squa_coor[:, 1] <= Ly/2) )

        return np.where( np.logical_and(cond_x, cond_y))[0]
    
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

                    index = self.check_circ_cond(pos = self.w_universe,
                                                 mid = mid,
                                                 r   = r)
                    
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
                    
                    index = self.check_square_cond(pos = self.w_universe,
                                                   Lx = Lx,
                                                   Ly = Ly,
                                                   mid = mid)


                    if not index.size > 0: continue
                    
                    self.d_coeffs[ index ] = self.domain_geometry[key][7]

                else:
                    raise ValueError("Currently I cannot handle the provided geometry.")

    #----------------------------------------------------------------------------------------------------------------------------------------
    #Metropolis
    @staticmethod
    def metropolis_decision(deltaE, RT):
        
        if deltaE <= 0: return True
        else:

            p_accept = np.min( [1, np.exp(- deltaE / RT ) ] )
            alpha    = np.random.rand(1)[0]

            if alpha < p_accept: return True
            else: return False
    
    @staticmethod
    def get_delta_E(x_new, x_old, f_true, N, forceconstant):

        """
        Calculation of the energy difference for changing the number of particles in domains.
        The energy difference is calculated via a simple spring potential.

        """

        #Calculate fraction of particles in domains
        f_old = x_old / N
        f_new = x_new / N

        #Calculate energy levels based on simple spring potential
        E_old = forceconstant * 0.5 * (f_old - f_true)**2 
        E_new = forceconstant * 0.5 * (f_new - f_true)**2

        #Calculate energy difference
        #<= 0 New Energy level is lower or equal -> accept
        # > 0 New Energy level is higher -> prop. reject

        #Unit: kJ/mol
        deltaE = ( E_new - E_old )

        return deltaE
    
#    @staticmethod
#    def get_delta_E_v2(x_new, x_old, r_true, N, forceconstant):
#
#        """
#        Calculation of the energy difference for changing the number of particles in domains.
#        The energy difference is calculated via a simple spring potential.
#
#        """
#
#        #Calculate fraction of particles in domains
#        fi_old, fo_old = x_old / N, (N-x_old) / N
#        fi_new, fo_new = x_new / N, (N-x_new) / N
#
#        r_old = fi_old / fo_old
#        r_new = fi_new / fo_new
#
#        #Calculate energy levels based on simple spring potential
#        E_old = (r_old - r_true)**2 
#        E_new = (r_new - r_true)**2 
#
#        #Calculate energy difference
#        #<= 0 New Energy level is lower or equal -> accept
#        # > 0 New Energy level is higher -> prop. reject
#
#        #Unit: kJ/mol
#        deltaE = forceconstant * 0.5 * ( E_new - E_old )
#
#        return deltaE

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
                deltaE = self.get_delta_E(x_old = x_old, x_new = x_new, N = self.N, f_true = self.target_fraction, forceconstant = self.fconstant)

            elif self.barrier != None:
                deltaE = self.barrier

            else: raise ValueError('Could not handle request!')
            
            #Entering is accepted
            if self.metropolis_decision(deltaE = deltaE, RT = self.RT): self.in_domains[ p_index ] = True
            #Entering is rejected
            else: index_after_metropolis.append( p_index ) #Particle will get reflected
        

        #INSIDE -> OUTSIDE

        #Decide for each particle indepently if it is allowed to enter the domain
        #Iterate over B.1
        for p_index in leaving_index:

            if self.target_fraction != None:
                x_old = self.in_domains.sum()     
                x_new = self.in_domains.sum() - 1
                deltaE = self.get_delta_E(x_old = x_old, x_new = x_new, N = self.N, f_true = self.target_fraction, forceconstant = self.fconstant)

            elif self.barrier != None:
                deltaE = self.barrier

            else: raise ValueError('Could not handle request!')
            
            #Leaving is accepted
            if self.metropolis_decision(deltaE = deltaE, RT = self.RT): self.in_domains[ p_index ] = False
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
            #For some steps the coordinates of the previous step are required
            #p_prev = self.w_universe - self.displace

            #p_prev[:, 0] %= self.size_x
            #p_prev[:, 1] %= self.size_y

            #assert np.allclose( self.w_universe_prev, p_prev ), 'Quick check!'
            #assert np.allclose( p_prev, self.w_universe_prev ), 'Quick check!'

            #---------------------------------------------------------------------------------------
            #Iterate over stored geometries
            for key, geometry in self.hard_boundaries_geometry.items():

                #Circular hard boundary
                if 'c' in key:
                    
                    #--------------------------------------------------------------------------------
                    #Calculate real distances
                    mid = self.hard_boundaries_geometry[key][0].reshape(1,2)
                    r   = self.hard_boundaries_geometry[key][1]
                    
                    index        = self.check_circ_cond(pos = self.w_universe     , mid = mid, r = r)
                    prev_index   = self.check_circ_cond(pos = self.w_universe_prev, mid = mid, r = r)
                    
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
                    norm, intersection = self.get_normals_circle(points      = self.w_universe[index],
                                                                 prev_points = self.w_universe_prev[index],
                                                                 d           = self.displace[index],
                                                                 mid         = mid,
                                                                 r           = r
                                                                 )
                    
                    new_pos, d_new = self.calc_reflection(intersection = intersection, p_in = self.w_universe[index], n = norm )
                    
                    if not np.all( index_in_domains == index[ self.check_circ_cond(pos = new_pos, mid = mid, r = r) ] ):
                        print('Error points are not in circle, but are expected to be in circle!')

                        print(index)
                        print(index[ self.check_circ_cond(pos = new_pos, mid = mid, r = r) ])

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

                        index      = self.check_square_cond(pos = self.w_universe     , Lx = Lx, Ly = Ly, mid = mid)
                        prev_index = self.check_square_cond(pos = self.w_universe_prev, Lx = Lx, Ly = Ly, mid = mid)
                        
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
                
                        new_pos, d_new = self.calc_reflection(intersection = intersection, p_in = self.w_universe[index], n = norm )
                
                        self.w_universe_prev[index] = intersection
                        self.displace[index]        = d_new
                        self.w_universe[index]      = new_pos
                        
                        counter += 1
                        
                        assert counter < 2, 'Weird'
                        
                
                else:
                    raise ValueError("Currently I cannot handle the provided geometry.")
            
    def calc_reflection(self, intersection, p_in, n):

        """
        Calculate reflection vector of vector d with normal n

        d := numpy.ndarray
            displacement vectors of particles that are reflected
        n := numpy.ndarray
            normal of geometry
        """

        #Vectorized form
        #d.shape = (Nr, 2)
        #n.shape = (n , 2)
        
        d_rest = p_in - intersection
        d_rest = self.apply_pbc_vector(d_rest)

        reflection_vector =  d_rest - 2 * np.sum(d_rest * n, axis = 1).reshape(-1, 1) * n
        
        reflection = intersection + reflection_vector

        reflection = self.apply_pbc( reflection.reshape(-1, 2) )

        #----------------------------------------------------
        #Test reflection
        pos_p_prev = -1 * d_rest
        pos_ref    = self.apply_pbc_vector(reflection_vector)

        dot_pp = np.sum(pos_p_prev  * n, axis = 1)
        dot_pr = np.sum(pos_ref     * n, axis = 1)

        dot_pp /= np.linalg.norm(pos_p_prev, axis = 1)
        dot_pr /= np.linalg.norm(pos_ref   , axis = 1)

        if not np.allclose(dot_pr, dot_pp) and not np.allclose(dot_pp, dot_pr):

            print(dot_pr)
            print(dot_pp)
            
            print('Intersection:')
            print(intersection)

            raise ValueError("In angle is not equal out angle")


        #----------------------------------------------------

        return reflection, reflection_vector
    

    #---------------------------------------------------------------------------------------------------------------------
    #Normal calculation

    def get_normals_circle(self, prev_points, points, d, mid, r):

        """
        Calculate normal vector of a two dimensional circle for a point, and the intersection with the circular boundary.

        The required normal vector here is the vector from the circle midpoint to the intersection point, where the particle gets reflected.

        The intersection is calculated solving the linear equation:

            || M + d * lambda || = r

        , where M is the directional vector between the circle midpoint and the previous position; d is the displacement vector between the previous
        and the current position; lambda is a scale factor for the displacement vector; and r is the radius of the circle.
        The linear equation has two solutions. Here, always the smaller value of lambda is assumed to be the required value.

        Parameters
        ----------
        points := numpy.ndarray
            coordinates of reflected points
        prev_points := numpy.ndarray
            previous coordinates of reflected points
        d := numpy.ndarray
            displace vector
        mid := numpy.ndarray
            circle midpoint
        r := float
            radius of the circle
        
        """

        #Previous and current positions -> both are wrapped!
        prev_points = prev_points.reshape(-1,2)
        points      = points.reshape(-1,2)

        #-------------------------------------
        #Solve linear equation
        M = (prev_points - mid)
        M = self.apply_pbc_vector(M)
        
        A = d[:, 0]**2 + d[:, 1]**2
        B = M[:, 0] * d[:, 0] + M[:, 1] * d[:, 1]
        C = M[:, 0]**2 + M[:, 1]**2

        lam1 = 2 * B + 2 * np.sqrt(B**2 - A * ( C - r**2 ) )
        lam1 /= 2 * A
        
        lam2 = 2 * B - 2 * np.sqrt(B**2 - A * ( C - r**2 ) )
        lam2 /= 2 * A

        lam = np.abs( np.vstack((lam1, lam2)) ).min(0)
        lam = lam.reshape(-1, 1)
        
        assert lam.shape[0] == points.shape[0], 'Lambda has the wrong shape'
        
        #Calculate intersection
        intersection = prev_points + lam * d
        intersection = self.apply_pbc( pos = intersection.reshape(-1, 2) )

        #-------------------------------------
        #Check if intersection is between old positions and new positions
        #The following code checks if the intersection is located on the shortest vector between the previous and the
        #current position based on the smallest image convention.

        #Vector from current positions to intersection
        p_inter = intersection - points
        p_inter = self.apply_pbc_vector(p_inter)
        
        #Vector from previous positions to intersection
        prev_inter = intersection - prev_points
        prev_inter = self.apply_pbc_vector(prev_inter)

        #Using squares and Pythagoras Theorem
        p_inter2    = np.sum(   (p_inter)**2, axis = 1)
        prev_inter2 = np.sum((prev_inter)**2, axis = 1)
        prev_d2     = np.sum(           d**2, axis = 1)

        check = p_inter2 + prev_inter2 + 2 * np.sqrt(p_inter2 * prev_inter2)

        if not np.all( np.abs(check - prev_d2) < 1E-8 ):

            print(lam1, lam2)
            print(lam)
            print('A',A)
            print('B',B)
            print('C',C)
            print('r', r)
            print('mid', mid)

            print("Something went wrong. Write debug files!")
            np.save(arr = prev_points, file = 'prev_positions.debug.npy')
            np.save(arr = points,      file = 'positions.debug.npy')
            np.save(arr = d,           file = 'displace.debug.npy')
            np.save(arr = d,           file = 'displace.debug.npy')

            raise ValueError('Intersection is not between positions!')

        #-------------------------------------
        #Check if intersection is on circle boundary
        #Calculate distance between intersection and circle midpoint
        dist2mid = intersection - mid
        dist2mid = self.apply_pbc_vector( dist2mid )
        dist2mid = np.linalg.norm(dist2mid, axis = 1)

        #Check for deviations
        if not np.all( np.abs(dist2mid - r) < 1E-8 ):

            #Take numerical inaccuracies into account
            points_dist      = np.linalg.norm( self.apply_pbc_vector(     points - mid), axis = 1) - r
            prev_points_dist = np.linalg.norm( self.apply_pbc_vector(prev_points - mid), axis = 1) - r

            #If current and previous positions are on different sides of the circle boundary any devation is accepted
            if np.all( (points_dist * prev_points_dist) < 0 ): pass
            else:

                print("Something went wrong. Write debug files!")
                np.save(arr = intersection, file = 'intersection.debug.npy')
                np.save(arr = prev_points, file = 'prev_positions.debug.npy')
                np.save(arr = points,      file = 'positions.debug.npy')
                np.save(arr = d,           file = 'displace.debug.npy')

                raise ValueError('Intersection is not on circle!')
        
        #-------------------------------------
        #Calculate normal vector
        norm = intersection - mid

        #Apply PBC
        norm = self.apply_pbc_vector(norm)
        
        #Normalize
        norm /= np.linalg.norm( norm, axis = 1).reshape(-1, 1)

        return norm, intersection

    @staticmethod
    def perpDot(a, b):
        
        a_vert = np.copy(a)
        a_vert = np.flip(a_vert, axis = 1)
        a_vert[:, 0] = -1 * a_vert[:, 0]

        return np.dot(a_vert, b)

    def get_point_between_points(self, p, d, e1, e2, Lx, Ly):

        c = e1 - p
        c = self.apply_pbc_vector(c)
        
        vert = (e2 - e1)
        vert = self.apply_pbc_vector(vert.reshape(1, -1) )[0]
        
        if vert.sum() == 0: vert = (e2 - e1)

        t = self.perpDot(c , vert) / self.perpDot(d, vert)

        intersection = p + t.reshape(-1, 1) * d

        #----------------------------------------------------------
        true_intersection = np.ones( p.shape[0] , dtype = bool)
        #Check if point is on vector

        e1_inter = intersection - e1
        e1_inter = self.apply_pbc_vector(e1_inter)
        
        e2_inter = intersection - e2
        e2_inter = self.apply_pbc_vector(e2_inter)

        e1_inter2 = np.sum((e1_inter)**2, axis = 1)
        e2_inter2 = np.sum((e2_inter)**2, axis = 1)
        e1_e22    = np.sum(vert**2)

        check = e1_inter2 + e2_inter2 + 2 * np.sqrt(e1_inter2 * e2_inter2)

        cond = (check - e1_e22) >= 1E-8
        
        true_intersection[ cond ] = False
        
        #---------------------------------------------------------
        #Check if point is on vector

        p_inter = intersection - p
        p_inter = self.apply_pbc_vector(p_inter)

        prev_pos = p-d

        prev_pos = self.apply_pbc(pos = prev_pos.reshape(-1, 2))
        
        prev_inter = intersection - prev_pos
        prev_inter = self.apply_pbc_vector(prev_inter)

        p_inter2    = np.sum(   (p_inter)**2, axis = 1)
        prev_inter2 = np.sum((prev_inter)**2, axis = 1)
        prev_p2      = np.sum(d**2, axis = 1)

        check = p_inter2 + prev_inter2 + 2 * np.sqrt(p_inter2 * prev_inter2)

        cond = np.abs(check - prev_p2) >= 1E-8
        
        true_intersection[ cond ] = False
        #---------------------------------------------------------
        
        cond = np.abs( np.sqrt(prev_inter2) ) <= 1E-8
        true_intersection[ cond ] = False


        """
        p_to_intersection = intersection - p
        p_to_intersection = self.apply_pbc_vector(p_to_intersection)
        p_to_intersection = np.abs(p_to_intersection)
 
        true_intersection[ p_to_intersection[:, 0] > (Lx / 2) ] = False
        true_intersection[ p_to_intersection[:, 1] > (Ly / 2) ] = False

        true_intersection[ intersection[:, 0] < 0 ] = False
        true_intersection[ intersection[:, 0] > self.size_x ] = False
        
        true_intersection[ intersection[:, 1] < 0 ] = False
        true_intersection[ intersection[:, 1] > self.size_y ] = False
        """
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
            
            norm = self.apply_pbc_vector(norm)
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

        #Storage arrays
        msd             = np.zeros(  lagtimes.shape[0],          dtype = np.float32)
        sd_per_particle = np.zeros( (lagtimes.shape[0], self.N), dtype = np.float32)

        u_storage_analysis = np.copy( self.u_storage[begin:stop, :, :] )

        assert nsteps_analysis == u_storage_analysis.shape[0], 'Not correct number of frames'

        #Evalulate lagtimes
        for i, lag in tqdm(enumerate(lagtimes), total = lagtimes.shape[0]):

            if i == 0: continue

            dr = u_storage_analysis[:-lag, :, :] - u_storage_analysis[lag:, :, :]

            sqdist = np.square(dr).sum(axis=-1)

            msd[i]             = sqdist.mean()
            sd_per_particle[i] = sqdist.mean(axis = 0)

        #Convert lagtimes array to physical time
        tau = np.float32(lagtimes)
        tau = tau * self.dt * self.nstxout

        return tau, msd, sd_per_particle
    
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
        for i, lag in tqdm(enumerate(lagtimes), total = lagtimes.shape[0]):

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

        


