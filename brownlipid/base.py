# ----PYTHON---- #
from typing import Union, Dict, Any

import numpy as np
from tqdm import tqdm 

class base:

    def __init__(
                           self,
                          size_x:                    float = 10.0,
                          size_y:                    float = 10.0,
                          size_z:                    float = 10.0,
                               N:                      int = 1000,
                             dim:                      int = 2,
                          nsteps:                      int = 5000,
                              dt:                    float = 0.1,
                         nstxout:                      int = 1,
                          nstchk:                      int = 1, 
                    base_d_coeff:                    float = 1.,
                         domains:                     dict = {},
                 hard_boundaries:                     dict = {},
                      metropolis:                     dict = {},
                            temp:                    float = 295,
                          output:                      str = 'output'

                ):

        
        self.size_x       = size_x
        self.size_y       = size_y
        self.size_z       = size_z
        self.N            = N
        self.dim          = dim
        self.nsteps       = nsteps
        self.dt           = dt
        self.nstxout      = nstxout
        self.nstchk       = nstchk
        self.base_d_coeff = base_d_coeff
        self.output       = output
        self.metropolis   = metropolis
        self.temp         = temp

        #Base checking
        assert self.nsteps >= self.nstxout, "Number of steps must be larger or equal than frequency of output (nstxout)!"
        assert self.nstchk >= self.nstxout, "Frequency of output (nstxout) must be smaller than frequency of checkpoints (nstchk)!"

        assert self.nsteps % self.nstxout == 0, "Frequency of output (nstxout) must be multiple of number of steps!"
        assert self.nsteps % self.nstchk  == 0, "Frequency of checkpoints (nstchk) must be multiple of number of steps!"
        assert self.nstchk % self.nstxout == 0, "Frequency of output (nstxout) must be multiple of frequency of checkpoints (nstchk)!"

        #----------------------------------------------------------------------------------------------------------------------------------------------
        #Init Domains with different diffusion coefficients

        if any( domains ):

            self.domain_geometry = {}

            for key, geometries in domains.items():

                if key == 'Circle':

                    for i, geometry in enumerate(geometries):

                        diff_coeff = geometry[0]
                        r          = geometry[1]
                        mx         = geometry[2]
                        my         = geometry[3]

                        self.domain_geometry[f"c{i}"] = [np.array([mx, my]), r, diff_coeff]

                elif key == 'Rectangle':
                    
                    for i, geometry in enumerate(geometries):

                        diff_coeff = geometry[0]
                        mx         = geometry[1]
                        my         = geometry[2]
                        Lx         = geometry[3]
                        Ly         = geometry[4]

                        assert Lx >= self.size_x / 2, 'Rectangle edge is larger than half the box length in x that can lead to problems!'
                        assert Ly >= self.size_y / 2, 'Rectangle edge is larger than half the box length in y that can lead to problems!'

                        #Generate rectangle edges clockwise
                        #Increase rectangle slightly in positive x- and y-direction
                        edges = []

                        edges.append( np.array([ mx - Lx/2 , my - Ly/2 ]) )
                        edges.append( np.array([ mx - Lx/2 , my + Ly/2 ]) )
                        edges.append( np.array([ mx + Lx/2 , my + Ly/2 ]) )
                        edges.append( np.array([ mx + Lx/2 , my - Ly/2 ]) )
                        
                        #domains[f"p{i}"] = [ np.array([ edge[0] % self.size_x, edge[1] % self.size_y ]) for edge in edges ]
                        
                        #Append the rest of the domain information
                        self.domain_geometry[f"p{i}"].append( Lx )
                        self.domain_geometry[f"p{i}"].append( Ly )
                        self.domain_geometry[f"p{i}"].append( np.array( [ mx, my ] ) )
                        self.domain_geometry[f"p{i}"].append(diff_coeff )
        
        else:
            self.domain_geometry = {}
        
        self.d_coeffs = np.ones( self.N ) * base_d_coeff
        
        #----------------------------------------------------------------------------------------------------------------------------------------------
        #Init Domains with hard boundary conditions

        if any( hard_boundaries ):

            hard_boundaries_geometry = {}

            self.total_area_domains = 0
            
            for key, geometries in hard_boundaries.items():

                if key == 'Circle':

                    for i, geometry in enumerate(geometries):

                        mx = geometry[1]
                        my = geometry[2]
                        r  = geometry[0]


                        hard_boundaries_geometry[f"c{i}"] = [np.array([mx,
                                                                       my]),
                                                                       r
                                                             ]

                        self.total_area_domains += np.pi * r**2

                elif key == 'Rectangle':
                    
                    for i, geometry in enumerate(geometries):

                        mx = geometry[0]
                        my = geometry[1]
                        Lx = geometry[2]
                        Ly = geometry[3]

                        #Generate rectangle edges clockwise
                        #Increase rectangle slightly in positive x- and y-direction
                        edges = []

                        edges.append( np.array([ mx - Lx/2 , my - Ly/2 ]) )
                        edges.append( np.array([ mx - Lx/2 , my + Ly/2 ]) )
                        edges.append( np.array([ mx + Lx/2 , my + Ly/2 ]) )
                        edges.append( np.array([ mx + Lx/2 , my - Ly/2 ]) )
                        
                        #hard_boundaries_geometry[f"p{i}"] = [ np.array([ edge[0] % self.size_x, edge[1] % self.size_y ]) for edge in edges ]
                        hard_boundaries_geometry[f"p{i}"] = [ np.array([ edge[0], edge[1] ]) for edge in edges ]
                        
                        hard_boundaries_geometry[f"p{i}"].append( Lx )
                        hard_boundaries_geometry[f"p{i}"].append( Ly )
                        hard_boundaries_geometry[f"p{i}"].append( np.array( [ mx, my ] ) )
                        
                        self.total_area_domains += Lx * Ly

            self.hard_boundaries_geometry = hard_boundaries_geometry

        else: self.hard_boundaries_geometry = {}
        
        #----------------------------------------------------------------------------------------------------------------------------------------------
        #Initialize for metropolis steps
        if any(self.metropolis):

            self.target_fraction = None
            self.barrier = None

            if   'Inside'     in self.metropolis.keys() and 'Barrier' not in self.metropolis.keys(): 

                self.target_fraction = self.metropolis['Inside']
                self.fconstant       = 1E8
            
            elif 'Inside' not in self.metropolis.keys() and 'Barrier'     in self.metropolis.keys(): self.barrier         = self.metropolis['Barrier']
            
            elif 'Inside'     in self.metropolis.keys() and 'Barrier'     in self.metropolis.keys(): raise ValueError('Can handle either Inside or Barrier for Metropolis. But not both!')

            else: raise ValueError('Can not handle metropolis request!')

            self.RT = 8.3145 * 1E-3 * self.temp

            """
            fin = self.metropolis['Inside']
            fout = 1. - fin

            self.metropolis['Ratio'] = fin / fout

            #Calculate force constant for spring potential


            sigma = self.fluctuations / 2

            self.fconstant = self.RT / (sigma)**2 


            print('Metropolis Algorithm was requested!')
            print('Temperature:', self.temp)
            print('Domain Ratio:', self.metropolis['Ratio'])
            #print('Target density outside domains:', self.metropolis['Outside'])
            print('Force constant (kJ/mol/nm^2):',  self.fconstant)
            print('Fluctuations:', self.fluctuations)
            """

        self.in_domains = np.zeros( self.N, dtype = bool ) 

        self.total_index = np.arange( self.N )

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Define functions
    
    def base_apply_pbc_vector(self, vec):

        #Apply PBC
        vec[0] = np.where(vec[0] >    self.size_x / 2, vec[0] - self.size_x, vec[0])
        vec[0] = np.where(vec[0] <= - self.size_x / 2, vec[0] + self.size_x, vec[0])

        vec[1] = np.where(vec[1] >    self.size_y / 2, vec[1] - self.size_y, vec[1])
        vec[1] = np.where(vec[1] <= - self.size_y / 2, vec[1] + self.size_y, vec[1])

        return vec


    
