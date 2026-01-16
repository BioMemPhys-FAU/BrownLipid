# ----PYTHON---- #
from typing import Union, Dict, Any

#Math
import numpy as np
from numba import jit

#Plots
import matplotlib.pyplot as plt

#Progress bar
from tqdm import tqdm

#Run information file
import time
import datetime
import json
import os
import platform

#from tests.utils_test import pbc_dim
#Own modules
from .base import base
from . import utils
from . import force
from . import reflection
from . import clean
from . import pressure

class Universe(base):

    def evolve(self):

        """
        Main Function
        
        In this function, the positions of the particles are initialized and then moved forward in time.
        For each step, the components of the displacement vector are randomly sampled from a gaussian distribution and are
        added to the actual position of the particles. If requested forces between particles emerging from a Lennard-Jones potential are considered. Periodic boundary conditions, domains with different diffusion coefficients, and hard boundaries 
        are taken into account.

        There are two "universe" objects that are considered during the setup and the evolution of the system:

            - A "wrapped" universe (self.w_universe) in which positions are bounded by the limits of the box (self.size_x, and self.size_y).
            - An "unwrapped" universe (self.u_universe) in which positions are not bounded by the limits of the box. This can be used for further analysis (e.g., MSD)

        To prevent a slow down of the simulation due to an increase in the stored trajectory, two methods are applied:

            - Frames are stored only every self.nstxout steps in w_storage/u_storage
            - Arrays w_storage/u_storage are written to disk every self.nstchk steps.

        Therefore self.nstchk // self.nstxout frames are stored in a single checkpoint file. Except the first frame, due to the initial frame.

        """

        #---------------------------------------------------------------------------------------------------------------------
        #Initialization Step

        #Save simulation start time
        start_timestamp = datetime.datetime.now()
        self.start_timestamp = start_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        self.start_time_diff = time.perf_counter()

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
        time_array = np.zeros(   nstchk_frames + 1          , dtype = np.float32 )
        f_inside   = np.zeros(  (nstchk_frames + 1, self.N), dtype = np.float32 )
        potEnergy  = np.zeros(   nstchk_frames + 1          , dtype = np.float32 )
        Virial     = np.zeros(   nstchk_frames + 1          , dtype = np.float32 )
        Pressure   = np.zeros(   nstchk_frames + 1          , dtype = np.float32 )
        ScalFac    = np.zeros(   nstchk_frames + 1          , dtype = np.float32 )
        Area       = np.zeros(   nstchk_frames + 1          , dtype = np.float32 )
        
        #Store initial frame
        w_storage[0] = self.w_universe
        u_storage[0] = self.u_universe
        
        #Store initial time
        time_array[0] = 0

        #Store initial fraction of particles in domains
        #f_inside[0] = self.in_domains

        #Store initial area
        Area[0] = self.area

        #This array is needed for further calculation. If no soft walls are used, it should always be 0.
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
            if self.softwall == True and len(self.pbc_dim[0]) < 2: self.force_from_wall = utils.apply_soft_wall_force(pos = self.w_universe, hard_wall = self.hard_wall, lj_cutoff = self.lj_cutoff, lj_A12 = self.lj_A12, lj_B6 = self.lj_B6)
            
            #Move particles in time
            self.forward_in_time()
            
            #Apply hard wall boundary conditions (if applicable)
            if   self.bounce_scale == None and len(self.pbc_dim[0]) < 2 and self.softwall == False: self.w_universe = self.apply_hard_wall(pos = self.w_universe) 
            elif self.bounce_scale != None and len(self.pbc_dim[0]) < 2 and self.softwall == False: self.w_universe = utils.apply_hard_wall_scale(pos = self.w_universe, d = self.displace, hard_wall = self.hard_wall, bounce_scale = self.bounce_scale)
            else: pass
            
            #Apply hard boundaries (if applicable)
            self.apply_hard_boundaries()

            #Apply periodic boundary conditions
            self.w_universe = self.apply_pbc(pos = self.w_universe)

            #Apply pressure coupling
            if self.pressure_coupling != False: self.pressure_coupling_step()



            #-----------------------------------------------------------------------------------------------------------------
            #Store particle positions

            #Unwrap self.w_universe and store the positions in self.u_universe
            #for j, size in enumerate([self.size_x, self.size_y]): self.u_universe[:, j] = self.w_universe[:, j] - np.floor( (self.w_universe[:, j] - self.u_universe[:, j]) / size + 0.5 ) * size

            #Unwrapping with TOR scheme
            for j, size in enumerate([self.size_x, self.size_y]):
                self.u_universe[:, j] = self.u_universe[:, j] + (self.w_universe[:, j] - self.w_universe_prev[:, j]) - np.floor( (self.w_universe[:, j] - self.w_universe_prev[:, j]) / size + 0.5 ) * size

            #Write current state of the universe into storage arrays
            if not i % self.nstxout: 

                #Calculate current time
                time_at_step = self.dt * i
                
                #--------------------------------------------------------------------------------------------------
                #Fill storage arrays

                #Current frame index in the current checkpoint file
                chk_frame_index = ( ( (i-1) // self.nstxout) % nstchk_frames ) + utils.heaviside(x = i, threshold = self.nstchk)
                
                w_storage[  chk_frame_index, :, : ] = self.w_universe
                u_storage[  chk_frame_index, :, : ] = self.u_universe
                
                time_array[ chk_frame_index ] = time_at_step

                f_inside[   chk_frame_index ] = self.in_domains

                potEnergy[  chk_frame_index ] = np.sum( self.pote )
                Virial[     chk_frame_index ] = np.sum( self.virial )

                if self.pressure_coupling != False:
                    Area[       chk_frame_index ] = self.area
                    Pressure[   chk_frame_index ] = self.p
                    ScalFac[    chk_frame_index ] = self.scal_fac

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

                    np.save( arr = Pressure  , file = self.output +   f"_Pressure.{chk_number.zfill(5)}" )
                    np.save( arr = ScalFac   , file = self.output +   f"_ScalFac.{chk_number.zfill(5)}" )
                    np.save( arr = Area      , file = self.output +   f"_Area.{chk_number.zfill(5)}" )
        
                    #Setup storage for positions -> Shape: (Number of frames in checkpoint file, Number of Particles, Number of dimensions)
                    w_storage  = np.zeros( ( nstchk_frames, self.N, 2 ), dtype = np.float32 )
                    u_storage  = np.zeros( ( nstchk_frames, self.N, 2 ), dtype = np.float32 )

                    #Setup storage for time steps -> Shape: (Number of frames in checkpoint file)
                    time_array = np.zeros( nstchk_frames         , dtype = np.float32 )
                    
                    f_inside = np.zeros(  (nstchk_frames, self.N), dtype = np.float32 )
        
                    potEnergy  = np.zeros( nstchk_frames         , dtype = np.float32 )
                    Virial     = np.zeros( nstchk_frames         , dtype = np.float32 )
                    Pressure   = np.zeros( nstchk_frames         , dtype = np.float32 )
                    Area       = np.zeros( nstchk_frames         , dtype = np.float32 )
                    ScalFac    = np.zeros( nstchk_frames         , dtype = np.float32 )
    
        self.last_frame = np.copy(self.w_universe)

        #Save simulation end time
        end_timestamp = datetime.datetime.now()
        self.end_timestamp = end_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        self.end_time_diff = time.perf_counter()

        #Save run file with run information
        self.save_run_info()

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
        init_pos = np.random.rand( self.N_only_lipids, 2 )
        
        #Scale the coordinates according to the box lengths
        for i, size in enumerate([self.size_x, self.size_y]): init_pos[:, i] *= size
        
        #----------------------------------------------------------------------------------------------------------
        #Clean restricted areas and distribute particles respecting pair interactions

        init_pos = clean.clean_domains(init_pos            = init_pos,
                                       geometry_collection = self.hard_boundaries_geometry,
                                       N                   = self.N_only_lipids,
                                       lj_sig              = self.lj_sig,
                                       pbc_dim             = self.pbc_dim,
                                       hard_wall           = self.hard_wall,
                                       soft_wall           = self.softwall,
                                       size_x              = self.size_x,
                                       size_y              = self.size_y,
                                       offsets             = self.offsets[:(self.N_only_lipids), :(self.N_only_lipids)])

        init_pos = np.vstack( (init_pos, self.domain_coords) )

        assert init_pos.shape[0] == self.N

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
        For each particle and for every dimension a random number from a standard normal distribution is drawn.
        The random number is then scaled by a factor taking a diffusion coefficient into account:

            sqrt( 2 * D * dt) (1)

        Equation 1 takes the diffusion coefficient (D) and the time step (dt) of the simulation.

        In this implementation D is an array (self.d_coeffs) that stores the diffusion coefficient of every particle.
        The diffusion coefficients can vary between the particles, e.g., if a particle is trapped in a domain.
        """
        
        #LJ + RANDOM
        if any(self.external_forces):
        
            diff_dt = (self.d_coeffs * self.dt).reshape(-1, 1)

            #Calculate the factor for every particle.
            factor = np.sqrt( 2 * diff_dt )

            #Calculate forces between particles
            lj_force, self.pp_pairlist, self.virial, self.pote = force.lennard_jones(frame         = self.frame,
                                                                                     nstlist       = self.lj_nstlist,
                                                                                     ref_pos       = self.w_universe,
                                                                                     conf_pos      = self.w_universe,
                                                                                     pbc_dim       = self.pbc_dim,
                                                                                     area          = self.area,
                                                                                     E_lrc_const   = self.E_lrc_const,
                                                                                     buffer_radius = self.lj_buffer,
                                                                                     vdw_cutoff    = self.lj_cutoff,
                                                                                     pairlist      = self.pp_pairlist,
                                                                                     A12           = self.lj_A12,
                                                                                     B6            = self.lj_B6,
                                                                                     offsets       = self.offsets)

            F = lj_force + self.force_from_wall

            #Calculate the displace vector
            self.displace = diff_dt * F / self.RT + factor * np.random.randn( self.N, 2 )

    #RANDOM
        else:

            diff_dt       = (self.d_coeffs * self.dt).reshape(-1, 1)

            #Calculate the factor for every particle.
            factor        = np.sqrt( 2 * diff_dt )
            
            self.displace = factor * np.random.randn( self.N, 2 )

            self.pote     = np.array([0])
            self.virial   = np.array([0])

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

    #-----------------------------------------------------------------------------------------------------------------------------
    #Pressure coupling

    def pressure_coupling_step(self):

        """
        Function to apply simple pressure coupling.
        Pressure of the system is calculated and used to calculate the rescaling factor for the pressure coupling.
        Factor is applied to rescale the whole system and change its volume respectively.
        Rescaling is applied equally in both dimensions!

        Formula based on Berendsen isotropic pressure coupling:
        linear:         scaling_factor = 1 + (compressibility * dt * (p - ref_p) / (3 * tau_p))
        exponential:    scaling_factor = (1 + (compressibility * dt * (p - ref_p) / tau_p)) ** (1/3)


        """

        self.scal_fac, self.p = pressure.scaling_factor(
                                            NkT             = self.NkT,
                                            area            = self.area,
                                            dt              = self.dt,
                                            virial          = self.virial,
                                            K_A             = self.K_A,
                                            ref_A           = self.ref_A,
                                            p_lrc_const     = self.p_lrc_const,
                                            ref_p           = self.ref_p,
                                            compressibility = self.compressibility,
                                            tau_p           = self.tau_p,
                                            thresh_p        = self.thresh_p)

        if self.frame % self.nstpcouple == 0:
            self.w_universe *= self.scal_fac

            self.size_x *= self.scal_fac
            self.size_y *= self.scal_fac

            self.area *= self.scal_fac**2

            #Update sizes in pbc_dim and hard_wall
            if len(self.pbc_dim[1]) == 1: self.pbc_dim[1][0] *= self.scal_fac
            if len(self.pbc_dim[1]) == 2:
                self.pbc_dim[1][0] *= self.scal_fac
                self.pbc_dim[1][1] *= self.scal_fac

            if len(self.hard_wall[1]) == 1: self.hard_wall[1][0] *= self.scal_fac
            if len(self.hard_wall[1]) == 2:
                self.hard_wall[1][0] *= self.scal_fac
                self.hard_wall[1][1] *= self.scal_fac

    #-----------------------------------------------------------------------------------------------------------------------------
    #Boundaries
    def apply_hard_boundaries(self):

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

    #--------------------------------------------------------------------------------------------------------------
    #Run information file
    def save_run_info(self):

        self.simulation_duration_sec = self.end_time_diff - self.start_time_diff

        if not self.simulation_duration_sec > 0: steps_per_second = 0
        else: steps_per_second = round(self.nsteps / self.simulation_duration_sec, 2)

        m, s = divmod(self.simulation_duration_sec, 60)
        h, m = divmod(m, 60)

        run_info = {
            'meta': {
                'working_directory': os.getcwd(),
                'start_time': self.start_timestamp,
                'end_time': self.end_timestamp,
            },
            'parameters': self.run_info_params,
            'performance': {
                'duration_seconds': round(self.simulation_duration_sec, 4),
                'duration': f'{int(h):d}h {int(m):02d}m {int(s):02d}s',
                'steps_per_second': steps_per_second
            }
        }

        filename = self.output + f"_run_info.json"

        with open(filename, 'w') as f:
            json.dump(run_info, f, indent=5)
        print(f'Saved run info in: {filename}')