# ----PYTHON---- #
from typing import Union, Dict, Any

import numpy as np
from tqdm import tqdm 

from . import utils

class base:

    def __init__(
                           self,
                          size_x:                    float = 10.0,
                          size_y:                    float = 10.0,
                               N:                      int = 1000,
                         pbc_dim:                      str = 'xy',
                          nsteps:                      int = 5000,
                              dt:                    float = 1.0,
                         nstxout:                      int = 1,
                          nstchk:                      int = 1, 
                    base_d_coeff:                    float = 1.,
                 hard_boundaries:                     dict = {},
                        softwall:                     bool = False,
                    bounce_scale:       Union[None, float] = None,
            checkpoint_structure:  Union[None, np.ndarray] = None,
                   hydrodynamics:                     bool = False,
                       viscosity:                    float = 1,
                  random_domains:                     bool = False,
                      metropolis:                     dict = {},
                 external_forces:                     dict = {},
                            temp:                    float = 298,
                          output:                      str = 'output'

                ):

        
        self.size_x               = size_x
        self.size_y               = size_y
        self.N                    = N
        self.nsteps               = nsteps
        self.dt                   = dt
        self.nstxout              = nstxout
        self.nstchk               = nstchk
        self.checkpoint_structure = checkpoint_structure
        self.base_d_coeff         = base_d_coeff
        self.output               = output
        self.metropolis           = metropolis
        self.hydrodynamics        = hydrodynamics
        self.bounce_scale         = bounce_scale
        self.viscosity            = viscosity
        self.external_forces      = external_forces
        self.temp                 = temp
        self.RT                   = 8.3145 * 1E-3 * self.temp
        self.d_coeffs             = np.repeat( self.base_d_coeff, self.N )

        #Base checking
        assert self.nsteps >= self.nstxout, "Number of steps must be larger or equal than frequency of output (nstxout)!"
        assert self.nstchk >= self.nstxout, "Frequency of output (nstxout) must be smaller than frequency of checkpoints (nstchk)!"

        assert self.nsteps % self.nstxout == 0, "Frequency of output (nstxout) must be multiple of number of steps!"
        assert self.nsteps % self.nstchk  == 0, "Frequency of checkpoints (nstchk) must be multiple of number of steps!"
        assert self.nstchk % self.nstxout == 0, "Frequency of output (nstxout) must be multiple of frequency of checkpoints (nstchk)!"
        
        #----------------------------------------------------------------------------------------------------------------------------------------------
        #PBC Handling
        pbc_index = []
        pbc_size  = []

        hard_wall_index = []
        hard_wall_size  = []

        for idx, size, dim in zip([0, 1], [self.size_x, self.size_y], ['x', 'y']):

            if dim in pbc_dim:
                if idx == 0: 
                    pbc_index.append(1)
                    pbc_size.append( self.size_y )
                else: 
                    pbc_index.append(0)
                    pbc_size.append( self.size_x )
            else:
                if idx == 0: 
                    hard_wall_index.append(1)
                    hard_wall_size.append( self.size_y )
                else: 
                    hard_wall_index.append(0)
                    hard_wall_size.append( self.size_x )
        
        self.pbc_dim   = (pbc_index, pbc_size)
        self.hard_wall = (hard_wall_index, hard_wall_size)                

        #----------------------------------------------------------------------------------------------------------------------------------------------
        #Init Domains with hard boundary conditions

        if any( hard_boundaries ):

            if softwall == True: hard_boundaries_geometry = {'Type': "Soft"}
            else: hard_boundaries_geometry = {'Type': "Hard"}

            self.total_area_domains = 0
            
            for key, geometries in hard_boundaries.items():

                if key == 'Circle':

                    for i, geometry in enumerate(geometries):

                        mx         = geometry[2]
                        my         = geometry[3]
                        r          = geometry[1]
                        diff_coeff = geometry[0]
                        prev_index = np.array([], dtype = np.int64) #Indices of particles inside circle in the previous frame
                        pairlist   = np.array([], dtype = np.int64)

                        mid = np.array([mx,my]).reshape(1, 2)

                        hard_boundaries_geometry[f"c{i}"] = [
                                                             mid,
                                                             r,
                                                             r ** 2,
                                                             prev_index,
                                                             diff_coeff,
                                                             pairlist
                                                             ]

                        self.total_area_domains += np.pi * r**2

                    if random_domains == True:

                        print("Random placement of domains requested!")
                        print("Attention! Previous coordinates will be overwritten!")

                        ran_mid = utils.distribute_domains_random_same_radius(number  = len(geometries),
                                                                              r       = r,
                                                                              size_x  = size_x,
                                                                              size_y  = size_y,
                                                                              pbc_dim = self.pbc_dim, output = self.output)

                        for i, ran_mid_i in enumerate(ran_mid): 
                            hard_boundaries_geometry[f"c{i}"][0] = ran_mid_i.reshape(1, 2)
                            hard_boundaries[key][i][2] = ran_mid_i[0]
                            hard_boundaries[key][i][3] = ran_mid_i[1]


                else: raise ValueError("Don't know geometry!")

            self.hard_boundaries_geometry = hard_boundaries_geometry
            self.hard_boundaries_final    = hard_boundaries

        else: self.hard_boundaries_geometry = {}
        
        #It is expected that no particle starts in a domain!
        self.in_domains = np.zeros( self.N, dtype = bool ) 
        
        #----------------------------------------------------------------------------------------------------------------------------------------------
        #Initialize for metropolis steps
        if any(self.metropolis):

            self.target_fraction = None
            self.barrier = None

            #Fraction of particles in domains is given
            if   'Inside'     in self.metropolis.keys() and 'Barrier' not in self.metropolis.keys(): 

                self.target_fraction = self.metropolis['Inside']
                self.fconstant       = 1E8
            
            #Barrier for particles to enter/leave domains is given
            elif 'Inside' not in self.metropolis.keys() and 'Barrier'     in self.metropolis.keys(): 

                self.barrier         = self.metropolis['Barrier']
                self.bltz_prob       = np.min([1.0, np.exp( - self.barrier / self.RT )])
            
            #Fraction and barrier is given -> Problem!
            elif 'Inside'     in self.metropolis.keys() and 'Barrier'     in self.metropolis.keys(): raise ValueError('Can handle either Inside or Barrier for Metropolis. But not both!')

            else: raise ValueError('Can not handle metropolis request!')

        
        #This parameter is needed later for initializing positions and must be defined also if no external forces are requested
        self.lj_sig = -1000

        if any(external_forces):

            #Pre-calculate lennard-jones parameters to sped up calculations
            self.lj_sig = external_forces['sigma']
            self.lj_eps = external_forces['epsilon']

            self.lj_A12 = 48 * self.lj_eps * self.lj_sig**12
            self.lj_B6  = 24 * self.lj_eps * self.lj_sig**6

            self.lj_nstlist = external_forces['nstlist']
            
            #Only squared sums are considerd later
            self.lj_cutoff  = external_forces['r_vdw']**2
            self.lj_buffer  = external_forces['r_list']**2
            
            #Only squared sums are considerd later
            self.lj_cutoff_org  = external_forces['r_vdw']
            self.lj_buffer_org  = external_forces['r_list']

        if self.hydrodynamics == True: 

            self.cutoff_2a = ( 2**(1/6) * external_forces['sigma'] )**2
            self.cutoff_a  =  2**(1/6) * external_forces['sigma'] / 2
            
            self.Dij = np.ones( (self.N, self.N, 2, 2), dtype = np.float32 ) * np.nan

            self.viscosity_scale =(1.380649 * self.temp) / ( self.viscosity * np.pi )

            print( self.viscosity_scale, (6 * self.cutoff_a), self.cutoff_a)

            self.Dij[range(self.N), range(self.N)] = np.eye(2)  * self.viscosity_scale / (6 * self.cutoff_a)

            print(np.eye(2)  * self.viscosity_scale / (6 * self.cutoff_a))


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Define functions
    
    def base_apply_pbc_vector(self, vec):

        #Apply PBC
        vec[0] = np.where(vec[0] >    self.size_x / 2, vec[0] - self.size_x, vec[0])
        vec[0] = np.where(vec[0] <= - self.size_x / 2, vec[0] + self.size_x, vec[0])

        vec[1] = np.where(vec[1] >    self.size_y / 2, vec[1] - self.size_y, vec[1])
        vec[1] = np.where(vec[1] <= - self.size_y / 2, vec[1] + self.size_y, vec[1])

        return vec


    
