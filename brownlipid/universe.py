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
from . import clean
from . import hydrodynamics

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
        time_array = np.zeros(   nstchk_frames + 1         , dtype = np.float32 )
        f_inside   = np.zeros(  (nstchk_frames + 1, self.N), dtype = np.float32 )
        potEnergy  = np.zeros(   nstchk_frames + 1         , dtype = np.float32 )
        Virial     = np.zeros(   nstchk_frames + 1         , dtype = np.float32 )
        
        #Store initial frame
        w_storage[0] = self.w_universe
        u_storage[0] = self.u_universe
        
        #Store initial time
        time_array[0] = 0

        #Store initial fraction of particles in domains
        f_inside[0] = self.in_domains

        self.force_from_wall = np.zeros((self.N, 2), dtype = np.float32)

        #---------------------------------------------------------------------------------------------------------------------
        #Main Iteration

        #Iterate over the requested number of steps.
        #It starts at 1, because time step 0 is the initial distribution of the points in the universe.
        #It ends at self.nsteps + 1 to ensure that the requested time is reached
        for i in tqdm(range(1, self.nsteps + 1) ):
            
            #-----------------------------------------------------------------------------------------------------------------
            #Evolve particle position

            self.frame = i
            
            #Calculate forces from softwall bouncing (if applicable)
            if self.softwall == True and len(self.pbc_dim[0]) < 2:

                if self.hydrodynamics == True: self.force_from_wall = utils.apply_soft_wall_force_hydro(pos = self.w_universe, hard_wall = self.hard_wall, lj_cutoff = self.lj_cutoff, lj_A12 = self.lj_A12, lj_B6 = self.lj_B6, Dij= self.Dij, N = self.N)
                else: self.force_from_wall = utils.apply_soft_wall_force(pos = self.w_universe, hard_wall = self.hard_wall, lj_cutoff = self.lj_cutoff, lj_A12 = self.lj_A12, lj_B6 = self.lj_B6)
            
            #Move particles in time
            self.forward_in_time()
            
            #Apply hard wall boundary conditions (if applicable)
            if   self.bounce_scale == None and len(self.pbc_dim[0]) < 2 and self.softwall == False: self.w_universe = self.apply_hard_wall(pos = self.w_universe) 
            elif self.bounce_scale != None and len(self.pbc_dim[0]) < 2 and self.softwall == False: self.w_universe = utils.apply_hard_wall_scale(pos = self.w_universe, d = self.displace, hard_wall = self.hard_wall, bounce_scale = self.bounce_scale)
            else: pass
            
            #Apply hard boundaries (if applicable)
            self.hard_boundaries()
            
            #Apply periodic boundary conditions
            self.w_universe = self.apply_pbc(pos = self.w_universe)

            #-----------------------------------------------------------------------------------------------------------------
            #Store particle positions

            #Unwrap self.w_universe and store the positions in self.u_universe
            for j, size in enumerate([self.size_x, self.size_y]): self.u_universe[:, j] = self.w_universe[:, j] - np.floor( (self.w_universe[:, j] - self.u_universe[:, j]) / size + 0.5 ) * size

            #Write current state of the universe into storage arrays
            if not i % self.nstxout: 

                #Calculate current time
                time = self.dt * i
                
                #--------------------------------------------------------------------------------------------------
                #Fill storage arrays

                #Current frame index in the current checkpoint file
                chk_frame_index = ( ( (i-1) // self.nstxout) % nstchk_frames ) + utils.heaviside(x = i, threshold = self.nstchk)
                
                w_storage[  chk_frame_index, :, : ] = self.w_universe
                u_storage[  chk_frame_index, :, : ] = self.u_universe
                
                time_array[ chk_frame_index ] = time

                f_inside[   chk_frame_index ] = self.in_domains 

                potEnergy[  chk_frame_index ] = self.pote.sum()
                Virial[     chk_frame_index ] = self.virial.sum()

                #If check pointing is requested then write current storage arrays to disk and renew them
                if not i % self.nstchk:
                
                    #Number of current check point
                    chk_number = str(i // self.nstchk)

                    #Write arrays to disk
                    np.save( arr = w_storage , file = self.output +   f"_wrap.{chk_number.zfill(5)}" )
                    np.save( arr = u_storage , file = self.output + f"_unwrap.{chk_number.zfill(5)}" )
                    
                    np.save( arr = time_array, file = self.output +   f"_time.{chk_number.zfill(5)}" )
                    
                    np.save( arr = f_inside  , file = self.output +   f"_f_inside.{chk_number.zfill(5)}" )

                    np.save( arr = potEnergy , file = self.output +   f"_potE.{chk_number.zfill(5)}" ) 
                    np.save( arr = Virial    , file = self.output +   f"_Virial.{chk_number.zfill(5)}" ) 
        
                    #Setup storage for positions -> Shape: (Number of frames in checkpoint file, Number of Particles, Number of dimensions)
                    w_storage  = np.zeros( ( nstchk_frames, self.N, 2 ), dtype = np.float32 )
                    u_storage  = np.zeros( ( nstchk_frames, self.N, 2 ), dtype = np.float32 )

                    #Setup storage for time steps -> Shape: (Number of frames in checkpoint file)
                    time_array = np.zeros( nstchk_frames         , dtype = np.float32 )
                    
                    f_inside = np.zeros(  (nstchk_frames, self.N), dtype = np.float32 )
        
                    potEnergy  = np.zeros( nstchk_frames         , dtype = np.float32 )
                    Virial     = np.zeros( nstchk_frames         , dtype = np.float32 )
    
        self.last_frame = np.copy(self.w_universe)
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

        if not isinstance(self.checkpoint_structure, type(None)): 

            assert isinstance(self.checkpoint_structure, type(np.array([]))), 'Checkpoint structure is not numpy.array type!'

            assert self.checkpoint_structure.shape == (self.N, 2), f'Checkpoint structure shape is not correct. Expected ({self.N}, 2), but got {self.checkpoint_structure.shape}!'

            return self.checkpoint_structure

        #----------------------------------------------------------------------------------------------------------
        #Standard Workflow
        #Sample particle positions from random uniform distribution
        init_pos = np.random.rand( self.N, 2 )
        
        #Scale the coordinates according to the box lengths
        for i, size in enumerate([self.size_x, self.size_y]): init_pos[:, i] *= size
        
        #----------------------------------------------------------------------------------------------------------
        #Clean restricted areas and distribute particles respecting pair interactions

        init_pos = clean.clean_domains(init_pos            = init_pos,
                                       geometry_collection = self.hard_boundaries_geometry,
                                       N                   = self.N,
                                       lj_sig              = self.lj_sig,
                                       pbc_dim             = self.pbc_dim,
                                       hard_wall           = self.hard_wall,
                                       soft_wall           = self.softwall,
                                       size_x              = self.size_x,
                                       size_y              = self.size_y,
                                       offsets             = self.offsets[:self.N, :self.N])

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

    #--------------------------------------------------------------------------------------------------------------

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
        
        w_universe_domains = np.vstack((self.w_universe, self.domain_coords))

        #LJ + RANDOM (no hydrodynamics)
        if any(self.external_forces) and not self.hydrodynamics:
        
            diff_dt = (self.d_coeffs * self.dt).reshape(-1, 1)

            #Calculate the factor for every particle.
            factor = np.sqrt( 2 * diff_dt )

            #Calculate forces between particles
            lj_force, self.pp_pairlist, self.virial, self.pote = force.lennard_jones(frame         = self.frame,
                                                                                     nstlist       = self.lj_nstlist,
                                                                                     ref_pos       = w_universe_domains,
                                                                                     conf_pos      = w_universe_domains,
                                                                                     pbc_dim       = self.pbc_dim,
                                                                                     buffer_radius = self.lj_buffer,
                                                                                     vdw_cutoff    = self.lj_cutoff,
                                                                                     pairlist      = self.pp_pairlist,
                                                                                     A12           = self.lj_A12,
                                                                                     B6            = self.lj_B6,
                                                                                     hydrodyn      = self.hydrodynamics,
                                                                                     offsets       = self.offsets)

            force = lj_force[:self.N] + self.force_from_wall

            #Calculate the displace vector
            self.displace = diff_dt * force / self.RT + factor * np.random.randn( self.N, 2 )

        #HYDRODYNAMICS_LJ + RANDOM
        elif any(self.external_forces) and self.hydrodynamics:

            #Calculate forces between particles
            lj_force, self.pp_pairlist, self.Dij = self.lennard_jones(frame           = self.frame,
                                                                      nstlist         = self.lj_nstlist,
                                                                      ref_pos         = w_universe_domains,
                                                                      conf_pos        = w_universe_domains,
                                                                      pbc_dim         = self.pbc_dim,
                                                                      buffer_radius   = self.lj_buffer,
                                                                      vdw_cutoff      = self.lj_cutoff,
                                                                      pairlist        = self.pp_pairlist,
                                                                      A12             = self.lj_A12,
                                                                      B6              = self.lj_B6,
                                                                      hydrodyn        = self.hydrodynamics,
                                                                      cutoff_a        = self.cutoff_a,
                                                                      cutoff_2a       = self.cutoff_2a,
                                                                      viscosity_scale = self.viscosity_scale,
                                                                      Dij             = self.Dij,
                                                                      offsets         = self.offsets)


            #Calculate forces between particles
            F = self.dt * (lj_force[:self.N] + self.force_from_wall) / self.RT

            #L = np.linalg.cholesky(self.Dij) 
            L = hydrodynamics.numba_cholesky(a = self.Dij).astype(np.float32)
            #Calculate the displace vector
            R = hydrodynamics.get_R(L = L, N = self.N_p_Domains, dt = self.dt)[:self.N]

            self.displace = F + R
        
        #RANDOM
        else:

            diff_dt = (self.d_coeffs * self.dt).reshape(-1, 1)

            #Calculate the factor for every particle.
            factor = np.sqrt( 2 * diff_dt )
            
            self.displace = factor * np.random.randn( self.N, 2 )

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
            
            pos[:, i] = np.where(pos[:, i] < 0   , ( -1 * pos[:, i]), pos[:, i])
            pos[:, i] = np.where(pos[:, i] > size, (2 * size - pos[:, i]), pos[:, i])

        return pos
    
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
        #not_index contains particle indices that leave the domain -> indices that are in prev_index but not in index
        #assume_unique=True, because no particle can be multiple times in the same domain
        not_index = np.setdiff1d(ar1 = prev_index, ar2 = index, assume_unique=True) #Return the unique values in ar1 that are not in ar2.
        B = self.in_domains[ not_index ]
        #Case B.1 -> TRUE:  Particle wants to leave domain -> METROPOLIS
        #Case B.2 -> FALSE: Particle stays outside domain -> Do not reflect -> That should only happen if the point in a previous iteration was allowed to leave the domain.
        

        #Sort particles for which Metropolis must be called
        entering_index = index[ ~A ]    #Case A.2
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
                
                #Calculate changing fraction
                x_old = self.in_domains.sum()     
                x_new = self.in_domains.sum() + 1
                deltaE = metropolis.get_delta_E(x_old = x_old, x_new = x_new, N = self.N, f_true = self.target_fraction, forceconstant = self.fconstant)
                
                #Entering is accepted
                if metropolis.metropolis_decision(deltaE = deltaE, RT = self.RT): self.in_domains[ p_index ] = True
                #Entering is rejected
                else: index_after_metropolis.append( p_index ) #Particle will get reflected

            elif self.barrier != None:
                deltaE = self.bltz_prob
            
                #Entering is accepted
                if metropolis.fast_metropolis_decision(deltaE = deltaE): self.in_domains[ p_index ] = True
                #Entering is rejected
                else: index_after_metropolis.append( p_index ) #Particle will get reflected

            else: raise ValueError('Could not handle request!')
            
        

        #INSIDE -> OUTSIDE

        #Decide for each particle indepently if it is allowed to enter the domain
        #Iterate over B.1
        for p_index in leaving_index:

            if self.target_fraction != None:
                
                #Calculate changing fraction
                x_old = self.in_domains.sum()     
                x_new = self.in_domains.sum() - 1

                deltaE = metropolis.get_delta_E(x_old = x_old, x_new = x_new, N = self.N, f_true = self.target_fraction, forceconstant = self.fconstant)
                
                #Leaving is accepted
                if metropolis.metropolis_decision(deltaE = deltaE, RT = self.RT): self.in_domains[ p_index ] = False
                #Leaving is rejected
                else: index_after_metropolis.append( p_index )

            elif self.barrier != None:
                deltaE = self.bltz_prob
                
                #Leaving is accepted
                if metropolis.fast_metropolis_decision(deltaE = deltaE): self.in_domains[ p_index ] = False
                #Leaving is rejected
                else: index_after_metropolis.append( p_index )

            else: raise ValueError('Could not handle request!')
            
    
        return np.array(index_after_metropolis), index[ A ]

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
        if not any(self.hard_boundaries_geometry) or self.hard_boundaries_geometry['Type'] == 'Soft': pass

        else:
            
            self.d_coeffs[:] = self.base_d_coeff

            #---------------------------------------------------------------------------------------
            #Iterate over stored geometries
            for key, geometry in self.hard_boundaries_geometry.items():

                if key == 'Type': continue

                #Circular hard boundary
                elif 'c' in key:
                
                    #--------------------------------------------------------------------------------
                    #Calculate real distances
                    mid        = self.hard_boundaries_geometry[key][0]
                    r          = self.hard_boundaries_geometry[key][1]
                    r_sq       = self.hard_boundaries_geometry[key][2]
                    prev_index = self.hard_boundaries_geometry[key][3]
                    diff_coeff = self.hard_boundaries_geometry[key][4]
                    #--------------------------------------------------------------------------------
                    org_index  = utils.check_circ_cond(pos = self.w_universe , mid = mid, r = r_sq, pbc_dim = self.pbc_dim)
                    
                    #This is a correct, but slow, way to get the indices of particles that were in the domain in the previous frame
                    #prev_index_old = utils.check_circ_cond(pos = self.w_universe_prev, mid = mid, r = r_sq, pbc_dim = self.pbc_dim)
                    #assert list(prev_index) == list(prev_index_old), f'Problem with new list {prev_index} and old list {prev_index_old}'
                    
                    #--------------------------------------------------------------------------------
                    #Perform Metropolis step if required
                    if any(self.metropolis): index, fix_inside_index = self.double_metropolis_scheme( index = org_index, prev_index = prev_index)
                    else: 
                        index = org_index 

                    #index is a list of particle indices that are reflected by the boundary 
                    
                    #--------------------------------------------------------------------------------
                    #Check if particles are reflected
                    if not index.size > 0:
                        #If no particles are reflected, the indices of the particles in the domains are passed on 
                        self.hard_boundaries_geometry[key][3] = org_index
                        continue

                    #--------------------------------------------------------------------------------
                    #Get indices of reflected particles that remain...
                    index_in_domains  = index[  self.in_domains[index] ] #...inside domains
                    index_out_domains = index[ ~self.in_domains[index] ] #...outside domains

                    #--------------------------------------------------------------------------------
                    #Get coordinates of reflected particles
                    pos_index         = self.w_universe[index]
                    prev_pos_index    = self.w_universe_prev[index]
                    displace_index    = self.displace[index]
                
                    #--------------------------------------------------------------------------------
                    #Calculate normals and intersection with the circular boundary
                    norm, intersection     = reflection.get_normals_circle(points     = pos_index,
                                                                          prev_points = prev_pos_index,
                                                                          d           = displace_index,
                                                                          mid         = mid,
                                                                          r           = r,
                                                                          pbc_dim     = self.pbc_dim)
                    
                    #Calculate new position after reflection
                    new_pos, new_displace  = reflection.calc_reflection(intersection = intersection, p_in = pos_index, n = norm, pbc_dim = self.pbc_dim, bounce_scale = self.bounce_scale )

                    #Update coordinates
                    self.w_universe[index] = new_pos

                    #--------------------------------------------------------------------------------
                    #There are rare (!) cases in which the particle is placed inside/outside the domain after the reflection
                    #The following lines handle with such edge cases

                    #Identify misplaced particles
                    real_index_in_domains  = index[     utils.check_circ_cond(pos = new_pos, mid = mid, r = r_sq, pbc_dim = self.pbc_dim) ]
                    real_index_out_domains = utils.get_elements_only_in_a(a = index, b = real_index_in_domains, assume_unique = True)
                    #real_index_out_domains = np.setdiff1d( ar1 = index, ar2 = real_index_in_domains, assume_unique = True)

                    #Ideally escaped and captured would be empty
                    escaped  = utils.get_elements_only_in_a( a= index_in_domains, b = real_index_in_domains, assume_unique = True)
                    captured = utils.get_elements_only_in_a( a= index_out_domains, b = real_index_out_domains, assume_unique = True)
                    #escaped  = np.setdiff1d(ar1 = index_in_domains,  ar2 = real_index_in_domains, assume_unique = True)
                    #captured = np.setdiff1d(ar1 = index_out_domains, ar2 = real_index_out_domains, assume_unique = True)

                    #The misplaced particles are just re-assigned 
                    self.in_domains[ escaped  ] = False
                    self.in_domains[ captured ] = True

                    not_reflected_index = utils.get_elements_only_in_a( a= org_index, b = index, assume_unique = True)
                    #not_reflected_index = np.setdiff1d( ar1 = org_index, ar2 = index, assume_unique =True)
                    self.hard_boundaries_geometry[key][3] =  np.union1d(ar1 = not_reflected_index, ar2 = real_index_in_domains)
                    
                    #Update diffusion coefficients
                    self.d_coeffs[ self.hard_boundaries_geometry[key][3] ] = diff_coeff

                else: raise ValueError("Currently I cannot handle the provided geometry.")
    

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Analysis part
    def load_data_unwrap(self, block = None, skip = 1):

        try:
            if self.u_storage.shape[0] == self.nsteps // self.nstxout // skip + 1: return 0
        except: pass

        if type(block) == type(None): block = np.arange( self.N )
        
        self.u_storage  = np.zeros( (0, len(block), 2), dtype = np.float32 )
        self.time_array = np.zeros( (0)               , dtype = np.float32 )

        for chk_number in range(1, self.nsteps // self.nstchk + 1 ):

            chk_number = str(chk_number)

            if chk_number == "1":

                self.u_storage  = np.vstack( (self.u_storage, np.load(self.output + f"_unwrap.{chk_number.zfill(5)}.npy")[::skip, block, :] ))
                self.time_array = np.append( self.time_array, np.load(self.output + f"_time.{chk_number.zfill(5)}.npy")[::skip] ) 

            else:

                data = np.load(self.output + f"_unwrap.{chk_number.zfill(5)}.npy")
                time = np.load(self.output + f"_time.{chk_number.zfill(5)}.npy")

                start_idx = np.where( (time * self.dt) % skip == 0 )[0][0]

                self.u_storage  = np.vstack( (self.u_storage, data[start_idx::skip, block, :] ))
                self.time_array = np.append( self.time_array, time[start_idx::skip]  ) 


        #Validation
        assert self.u_storage.shape[0] == self.nsteps // self.nstxout // skip + 1, "Number of frames is not correct!"

        assert np.allclose( np.diff( self.time_array ), self.dt * self.nstxout * skip), "Time step distance is not as expected!"
        assert np.allclose( self.dt * self.nstxout * skip, np.diff( self.time_array ) ), "Time step distance is not as expected!"

        return 1
    
    def load_data_wrap(self, block = None, skip = 1):

        try:
            if self.w_storage.shape[0] == self.nsteps // self.nstxout // skip + 1: return 0
        except: pass

        if block == None: block = np.arange( self.N )
        
        self.w_storage  = np.zeros( (0, len(block), 2), dtype = np.float32 )
        self.time_array = np.zeros( (0)               , dtype = np.float32 )

        for chk_number in range(1, self.nsteps // self.nstchk + 1 ):

            chk_number = str(chk_number)

            if chk_number == "1":

                self.w_storage  = np.vstack( (self.w_storage, np.load(self.output + f"_wrap.{chk_number.zfill(5)}.npy")[::skip, block, :] ))
                self.time_array = np.append( self.time_array, np.load(self.output + f"_time.{chk_number.zfill(5)}.npy")[::skip] ) 

            else:

                data = np.load(self.output + f"_wrap.{chk_number.zfill(5)}.npy")
                time = np.load(self.output + f"_time.{chk_number.zfill(5)}.npy")

                start_idx = np.where( (time * self.dt) % skip == 0 )[0][0]

                self.w_storage  = np.vstack( (self.w_storage, data[start_idx::skip, block, :] ))
                self.time_array = np.append( self.time_array, time[start_idx::skip]  ) 

        print(f"Check {chk_number}: {self.time_array[-1]}")

        #Validation
        assert self.w_storage.shape[0] == ( self.nsteps // self.nstxout // skip + 1), f"Number of frames is not correct! Expected: {self.nsteps // self.nstxout // skip + 1} Got: {self.w_storage.shape[0]}"

        assert np.allclose( np.diff( self.time_array ), self.dt * self.nstxout * skip), "Time step distance is not as expected!"
        assert np.allclose( self.dt * self.nstxout * skip, np.diff( self.time_array ) ), "Time step distance is not as expected!"

        return 1


    def mean_square_displacement(self, begin = 0, stop = None, fft = True, block = None):

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

        if type(block) == type(None): block = np.arange( self.N )
        
        #Convert time to frames
        begin = int( np.round( begin / self.dt / self.nstxout ) )
        stop  = int( np.round( stop  / self.dt / self.nstxout ) )
        
        #Number of steps for analysis
        nsteps_analysis = stop - begin

        print(f"Calculating MSD...")
        print(f"Start: Frame {begin}")
        print(f"Stop : Frame {stop}")
        print("")

        #Load data
        self.load_data_unwrap(block = block)

        u_storage_analysis = np.copy( self.u_storage[begin:stop, :, :] )

        assert nsteps_analysis == u_storage_analysis.shape[0], 'Not correct number of frames'

        lagtimes = np.arange(0, nsteps_analysis, 1)
        
        print("Start analysis...")
        if fft == True: msd, sd_per_particle = utils.MSD_fft_ax(pos = u_storage_analysis)
        else: msd, sd_per_particle = utils.evaluate_lagtimes(pos = u_storage_analysis, lagtimes = lagtimes, N = u_storage_analysis.shape[1] )
        
        #Convert lagtimes array to physical time
        tau = lagtimes.astype( np.float32 )
        tau = tau * self.dt * self.nstxout

        return tau, msd, sd_per_particle
    
    def get_rdf(self, exp_density, begin = 0, stop = None, skip = None, block = None, r_max = 5, binwidth = 0.5):

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

        if block == None: block = np.arange( self.N )
        if skip  == None: skip  = self.dt / self.nstxout
        
        #Convert time to frames
        skip  = int( np.round( skip  / self.dt / self.nstxout ) )
        begin = int( np.round( begin / self.dt / self.nstxout ) ) // skip
        stop  = int( np.round( stop  / self.dt / self.nstxout ) ) // skip
        
        #Number of steps for analysis
        nsteps_analysis = (stop - begin)

        print(f"Calculating RDF...")
        print(f"Start: Frame {begin}")
        print(f"Stop : Frame {stop} ")
        print(f"Skip : Frame {skip} ") 
        print("")

        #Load data
        self.load_data_wrap(skip = skip)

        w_storage_analysis = np.copy( self.w_storage[begin:stop, :, :] )

        assert nsteps_analysis == w_storage_analysis.shape[0], 'Not correct number of frames'

        binmids, rdf, cdf = utils.self_rdf(exp_density = exp_density, pos = w_storage_analysis, pbc_dim = self.pbc_dim, r_max = r_max, binwidth = binwidth)
        
        return binmids, rdf.mean(0), cdf.mean(0)
    
    def mean_square_displacement_1d(self, direction, skip, begin = 0, stop = None, max_lag = "max", fft=True):

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
        skip  = int( np.round( skip  / self.dt / self.nstxout) )

        assert (self.nstxout % skip) == 0, 'Skip must be a multiple of nstxout'
        
        #Number of steps for analysis
        nsteps_analysis = stop - begin

        #Lag times at which MSD is evaluated
        if max_lag == "max": lagtimes = np.arange(0, nsteps_analysis, skip)
        else:

            max_lag = int( np.round( begin / self.dt ) )

            lagtimes = np.arange(0, max_lag, skip)

        print(f"Calculating MSD...")
        print(f"Start: Frame {begin}")
        print(f"Stop : Frame {stop}")
        print(f"Skip : Frame {skip}")
        print(f"Frames: {nsteps_analysis}")
        print("")


        #Load data
        self.load_data_unwrap()

        #Storage arrays
        msd             = np.zeros(  lagtimes.shape[0],          dtype = np.float32)
        sd_per_particle = np.zeros( (lagtimes.shape[0], self.N), dtype = np.float32)

        u_storage_analysis = np.copy( self.u_storage[begin:stop, :, direction] )

        assert nsteps_analysis == u_storage_analysis.shape[0], 'Not correct number of frames'

        #msd, sd_per_particle = utils.evaluate_lagtimes(pos = u_storage_analysis, lagtimes = lagtimes, N = u_storage_analysis.shape[1] )
        if fft == True: msd, sd_per_particle = utils.MSD_fft_ax(pos = u_storage_analysis.reshape(-1, self.N, 1))
        else: msd, sd_per_particle = utils.evaluate_lagtimes(pos = u_storage_analysis, lagtimes = lagtimes, N = u_storage_analysis.shape[1] )

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

        #sd_per_particle = sqdist.flatten() #sqdist.mean(axis = 0)
        sd_per_particle = sqdist# np.mean(sqdist, axis = 0)#.mean(axis = 0)

        print(f'Analysis from {begin * self.dt * self.nstxout / 1000 / 1000} to {stop * self.dt * self.nstxout / 1000 / 1000}')
        print(f'MSD Distribution at {lag * self.dt * self.nstxout / 1000 / 1000} ms') 

        return sd_per_particle
    
    def mean_square_displacement_distr_vectors(self, tau, grid_spacing = 1, begin = 0, stop = None):
        
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

        dr = u_storage_analysis[lag:, :, :] - u_storage_analysis[:-lag, :, :]

        dr = dr[::1]

        storage = []

        grid, idx_grid, full_grid, nx, ny = utils.generate_grid(size_x = self.size_x,
                                                                size_y = self.size_y,
                                                                grid_spacing = grid_spacing)

        for i, dr_i in tqdm( enumerate(dr), total = dr.shape[0] ):

            storage_i = utils.vector_field(pos = self.u_storage[:-lag, :, :][i], 
                                           displacement = dr_i,
                                           grid = grid,
                                           idx_grid = idx_grid,
                                           nx = nx, 
                                           ny = ny,
                                           pbc_dim = self.pbc_dim)


            storage.append(storage_i) 


        #-------------------------------------------------------------------------------------------------------------------
        #Plotting

        cyberpunk_theme = {
                           'axes.edgecolor': 'white',
                           'axes.facecolor': '#1d1f21',
                           'axes.labelcolor': 'white',
                           'axes.titlecolor': 'white',
                           'figure.facecolor': '#1d1f21',
                           'xtick.color': 'white',
                           'ytick.color': 'white',
                           'text.color': 'white',
                           'grid.color': 'gray',
                           'grid.linestyle': ':'
                          }

        # Set cyberpunk theme as default
        plt.style.use(cyberpunk_theme)

        #Cyperpunk
        fig, ax = plt.subplots()

        for spine in ['left', 'right', 'bottom', 'top']:
            ax.spines[spine].set_color('#66ccff')
            ax.spines[spine].set_linewidth(2)
        ax.tick_params(axis='x', colors='#66ccff')
        ax.tick_params(axis='y', colors='#66ccff')


        storage = np.array(storage)
        
        storage_mean = np.nanmean( storage, axis = 0)

        ax.quiver(full_grid[:, :, 0], full_grid[:, :, 1], storage_mean[:,:,0], storage_mean[:,:,1], scale = 5.0,
                   color = 'hotpink',
                   angles='xy',
                   scale_units='xy',
                   units = 'xy',
                   pivot = 'mid', alpha = 1.)#, cmap='magma', C = arrow_length)

        ax.set_aspect('equal')
        
        plt.savefig(f"{self.output}_vector_field.png", dpi = 300)
        
        plt.close()
        #------------------------------------------------------------
        
        #Cyperpunk
        fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

        ax.set_theta_zero_location(loc = 'N')
        ax.set_theta_direction(-1)

        #for spine in ['left', 'right', 'bottom', 'top']:
        #    ax.spines[spine].set_color('#66ccff')
        #    ax.spines[spine].set_linewidth(2)
        #ax.tick_params(axis='x', colors='#66ccff')
        #ax.tick_params(axis='y', colors='#66ccff')

        arrow_length = np.sqrt(np.nansum(storage**2, axis = -1))

        storage = storage / arrow_length[:, :, :, np.newaxis]

        x_angle_distr = np.clip(a = np.sum(storage * np.array([1., 0.]), axis = -1), a_min = -1, a_max = 1)
        x_angle_distr = np.arccos( x_angle_distr )

        x_angle_distr = np.where(storage[:, :, :, 1] < 0, (2*np.pi - x_angle_distr), x_angle_distr)

        y_angle_distr = np.clip(a = np.sum(storage * np.array([0., 1.]), axis = -1), a_min = -1, a_max = 1)
        y_angle_distr = np.arccos( y_angle_distr )

        print(storage.shape)
        
        y_angle_distr = np.where(storage[:, :, :, 0] < 0, (2*np.pi - y_angle_distr), y_angle_distr)

        
        ax.hist(x_angle_distr.flatten(), bins = np.linspace(0, 2*np.pi, 51), histtype = 'step', label = 'x-axis', density = True, color = 'hotpink')
        ax.hist(y_angle_distr.flatten(), bins = np.linspace(0, 2*np.pi, 51), histtype = 'step', label = 'y-axis', density = True, color = '#66ccff')

        #plt.legend()

        plt.savefig(f"{self.output}_vector_field_angles.png", dpi = 300)


        print(f'Analysis from {begin * self.dt * self.nstxout / 1000 / 1000} to {stop * self.dt * self.nstxout / 1000 / 1000}')
        print(f'MSD Distribution at {lag * self.dt * self.nstxout / 1000 / 1000} ms') 

    
    @staticmethod
    def mean_square_displacement_fit(tau, msd, dim = 2, begin = 0, stop = None):

        if stop == None: stop = tau[-1]

        mask = np.logical_and( begin <= tau, tau <= stop)

        assert np.all( ~mask ) == False, 'Required range is not found in tau'

        fit_tau = tau[mask]
        fit_msd = msd[mask]

        #tau -> ns
        #msd -> nm2
        DiffCoeff, Intercept = np.polyfit(x = fit_tau, y = fit_msd, deg = 1)

        DiffCoeff /= (2 * dim)

        #DiffCoeff -> nm2/ns -> um2/ms
        #Intercept -> nm2

        return DiffCoeff, Intercept

    @staticmethod
    def plot_log_histogram_mean_square_displacement_distr(ax, sd_per_particle, label, lo_limit = 1E-4, up_limit = 1.0, nbins = 51, color = 'red'):

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

        ax.hist(sd_per_particle, 
                density=True, 
                bins = np.logspace(np.log10(lo_limit),np.log10(up_limit), nbins),
                histtype = 'stepfilled',
                alpha = 0.3,
                edgecolor = color,
                color = color,
                label = label
                )

        #Scale
        ax.set_xscale('log')

        #Label
        #plt.ylabel('Number of Trajectories')
        #plt.xlabel(r'MSD / $\mu$m$^2$')
    
