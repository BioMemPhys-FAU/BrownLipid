# ----PYTHON---- #

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
        potEnergy  = np.zeros(   nstchk_frames + 1          , dtype = np.float32 )
        Virial     = np.zeros(   nstchk_frames + 1          , dtype = np.float32 )
        Pressure   = np.zeros(   nstchk_frames + 1          , dtype = np.float32 )
        ScalFac    = np.zeros(   nstchk_frames + 1          , dtype = np.float32 )
        Area       = np.zeros(   nstchk_frames + 1          , dtype = np.float32 )
        Velocity   = np.zeros(   (nstchk_frames + 1, self.N, 2), dtype = np.float32 )
        
        #Store initial frame
        w_storage[0] = self.w_universe
        u_storage[0] = self.u_universe
        
        #Store initial time
        time_array[0] = 0

        #Store initial area
        Area[0] = self.area

        #This array is needed for further calculation. If no soft walls are used, it should always be 0.
        self.force_from_wall = np.zeros((self.N, 2), dtype = np.float32)

        if self.langevin_dynamics != False:

            #Generate initial momenta by sampling from Maxwell-Boltzmann distribution
            self.momentum = utils.generate_initial_momenta(N = self.N, mass_per_lip = self.mass, temp = self.temp)

            #Store initial velocity
            Velocity[0] = 1e3 * self.momentum / self.mass

            # Calculate forces between particles for first half-step kick
            self.lj_force, self.pp_pairlist, self.virial, self.pote = force.lennard_jones(frame=1,
                                                                                          nstlist=self.lj_nstlist,
                                                                                          ref_pos=self.w_universe,
                                                                                          conf_pos=self.w_universe,
                                                                                          pbc_dim=self.pbc_dim,
                                                                                          area=self.area,
                                                                                          E_lrc_const=self.E_lrc_const,
                                                                                          buffer_radius=self.lj_buffer,
                                                                                          vdw_cutoff=self.lj_cutoff,
                                                                                          pairlist=self.pp_pairlist,
                                                                                          A12=self.lj_A12,
                                                                                          B6=self.lj_B6,
                                                                                          offsets=self.offsets)

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

            #Apply periodic boundary conditions
            self.w_universe = self.apply_pbc(pos = self.w_universe)

            #Apply pressure coupling
            if self.pressure_coupling != False: self.pressure_coupling_step()

            #-----------------------------------------------------------------------------------------------------------------
            #Store particle positions

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

                potEnergy[  chk_frame_index ] = self.pote
                Virial[     chk_frame_index ] = np.sum( self.virial )

                if self.pressure_coupling != False:
                    Area[       chk_frame_index ] = self.area
                    Pressure[   chk_frame_index ] = self.p
                    ScalFac[    chk_frame_index ] = self.scal_fac

                if self.langevin_dynamics != False: Velocity[ chk_frame_index ] = self.momentum * 1e3 / self.mass

                #If check pointing is requested then write current storage arrays to disk and renew them
                if not i % self.nstchk:
                
                    #Number of current check point
                    chk_number = str(i // self.nstchk)

                    #Write arrays to disk
                    np.save( arr = w_storage , file = self.output +   f"_wrap.{chk_number.zfill(5)}" )
                    np.save( arr = u_storage , file = self.output +   f"_unwrap.{chk_number.zfill(5)}" )
                    
                    np.save( arr = time_array, file = self.output +   f"_time.{chk_number.zfill(5)}" )

                    np.save( arr = potEnergy , file = self.output +   f"_potE.{chk_number.zfill(5)}" ) 
                    np.save( arr = Virial    , file = self.output +   f"_Virial.{chk_number.zfill(5)}" )

                    np.save( arr = Pressure  , file = self.output +   f"_Pressure.{chk_number.zfill(5)}" )
                    np.save( arr = ScalFac   , file = self.output +   f"_ScalFac.{chk_number.zfill(5)}" )
                    np.save( arr = Area      , file = self.output +   f"_Area.{chk_number.zfill(5)}" )

                    np.save(arr = Velocity   , file = self.output +   f"_Velocity.{chk_number.zfill(5)}" )
        
                    #Setup storage for positions -> Shape: (Number of frames in checkpoint file, Number of Particles, Number of dimensions)
                    w_storage  = np.zeros( ( nstchk_frames, self.N, 2 ), dtype = np.float32 )
                    u_storage  = np.zeros( ( nstchk_frames, self.N, 2 ), dtype = np.float32 )

                    #Setup storage for time steps -> Shape: (Number of frames in checkpoint file)
                    time_array = np.zeros( nstchk_frames         , dtype = np.float32 )

                    potEnergy  = np.zeros( nstchk_frames         , dtype = np.float32 )
                    Virial     = np.zeros( nstchk_frames         , dtype = np.float32 )
                    Pressure   = np.zeros( nstchk_frames         , dtype = np.float32 )
                    Area       = np.zeros( nstchk_frames         , dtype = np.float32 )
                    ScalFac    = np.zeros( nstchk_frames         , dtype = np.float32 )
                    Velocity   = np.zeros( (nstchk_frames, self.N, 2), dtype = np.float32 )
    
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
        init_pos = np.random.rand( self.N, 2 )
        
        #Scale the coordinates according to the box lengths
        for i, size in enumerate([self.size_x, self.size_y]): init_pos[:, i] *= size
        
        #----------------------------------------------------------------------------------------------------------
        #Clean restricted areas and distribute particles respecting pair interactions

        init_pos = clean.clean_domains(init_pos            = init_pos,
                                       N                   = self.N,
                                       lj_sig              = self.lj_sig,
                                       sigma_matrix        = self.sigma_matrix,
                                       pbc_dim             = self.pbc_dim,
                                       hard_wall           = self.hard_wall,
                                       soft_wall           = self.softwall,
                                       size_x              = self.size_x,
                                       size_y              = self.size_y,
                                       offsets             = self.offsets[:self.N, :self.N])

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

        #Langevin Dynamics Integrator - BAOAB scheme
        if self.langevin_dynamics:

            #Half-step kick (B) with previously calculated forces
            self.momentum += self.dt * self.lj_force * 1e3 / 2 # kg * nm / (ns * mol)

            #Half-step drift (A)
            w_universe_half = self.w_universe + self.dt * self.momentum * 1e3 / (2 * self.mass) # nm

            #Full-step frictional and random forces (O)
            self.momentum = self.c1 * self.momentum + self.c3 * np.random.normal(0, 1, size=(self.N, 2)) # kg * nm / (ns * mol)

            # Store previous positions
            self.w_universe_prev = np.copy(self.w_universe)

            #Half-step drift (A)
            self.w_universe = w_universe_half + self.dt * self.momentum * 1e3 / (2 * self.mass) # nm

            #Calculate forces between particles
            self.lj_force, self.pp_pairlist, self.virial, self.pote = force.lennard_jones(frame=self.frame + 1,
                                                                                     nstlist=self.lj_nstlist,
                                                                                     ref_pos=self.w_universe,
                                                                                     conf_pos=self.w_universe,
                                                                                     pbc_dim=self.pbc_dim,
                                                                                     area=self.area,
                                                                                     E_lrc_const=self.E_lrc_const,
                                                                                     buffer_radius=self.lj_buffer,
                                                                                     vdw_cutoff=self.lj_cutoff,
                                                                                     pairlist=self.pp_pairlist,
                                                                                     A12=self.lj_A12,
                                                                                     B6=self.lj_B6,
                                                                                     offsets=self.offsets)

            #Half-step kick (B)
            self.momentum += self.dt * self.lj_force * 1e3 / 2 # kg * nm / (ns * mol)

        #LJ + RANDOM
        elif any(self.external_forces):
            diff_dt = (self.d_coeffs * self.dt).reshape(-1, 1)

            # Calculate the factor for every particle.
            factor = np.sqrt(2 * diff_dt)

            # Calculate forces between particles
            lj_force, self.pp_pairlist, self.virial, self.pote = force.lennard_jones(frame=self.frame,
                                                                                     nstlist=self.lj_nstlist,
                                                                                     ref_pos=self.w_universe,
                                                                                     conf_pos=self.w_universe,
                                                                                     pbc_dim=self.pbc_dim,
                                                                                     area=self.area,
                                                                                     E_lrc_const=self.E_lrc_const,
                                                                                     buffer_radius=self.lj_buffer,
                                                                                     vdw_cutoff=self.lj_cutoff,
                                                                                     pairlist=self.pp_pairlist,
                                                                                     A12=self.lj_A12,
                                                                                     B6=self.lj_B6,
                                                                                     offsets=self.offsets)

            F = lj_force + self.force_from_wall

            #Calculate the displace vector
            self.displace = diff_dt * F / self.RT + factor * np.random.randn(self.N, 2)

            #Store previous positions
            self.w_universe_prev = np.copy(self.w_universe)

            #Move particles
            self.w_universe += self.displace

        #RANDOM
        else:

            diff_dt       = (self.d_coeffs * self.dt).reshape(-1, 1)

            #Calculate the factor for every particle.
            factor        = np.sqrt( 2 * diff_dt )

            self.displace = factor * np.random.randn( self.N, 2 )

            self.pote     = 0
            self.virial   = np.array([0])

            #Store previous positions
            self.w_universe_prev = np.copy( self.w_universe )

            #Move particles
            self.w_universe += self.displace

    #-----------------------------------------------------------------------------------------------------------------------------
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

    #-----------------------------------------------------------------------------------------------------------------------------
    #Pressure coupling

    def pressure_coupling_step(self):

        """
        Function to apply simple pressure coupling.
        Pressure of the system is calculated and used to calculate the rescaling factor for the pressure coupling.
        Factor is applied to rescale the whole system and change its volume respectively.
        Rescaling is applied equally in both dimensions!

        Formula based on Berendsen isotropic pressure coupling:
        scaling_factor = (1 + (compressibility * dt * (p - ref_p) / tau_p)) ** (1/2)


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