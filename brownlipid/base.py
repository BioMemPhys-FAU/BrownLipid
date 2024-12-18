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

        #Base checking
        assert self.nsteps >= self.nstxout, "Number of steps must be larger or equal than frequency of output (nstxout)!"
        assert self.nstchk >= self.nstxout, "Frequency of output (nstxout) must be smaller than frequency of checkpoints (nstchk)!"

        assert self.nsteps % self.nstxout == 0, "Frequency of output (nstxout) must be multiple of number of steps!"
        assert self.nsteps % self.nstchk  == 0, "Frequency of checkpoints (nstchk) must be multiple of number of steps!"
        assert self.nstchk % self.nstxout == 0, "Frequency of output (nstxout) must be multiple of frequency of checkpoints (nstchk)!"

        #Init domains
        if any( domains ):

            for key, geometries in domains.items():

                if key == 'Circle':

                    for i, geometry in enumerate(geometries):

                        domains[f"c{i}"] = [np.array([geometry[1], geometry[2]]), geometry[0]]

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
                        
                        domains[f"p{i}"] = [ np.array([ edge[0] % self.size_x, edge[1] % self.size_y ]) for edge in edges ]
                        
                        domains[f"p{i}"].append( Lx )
                        domains[f"p{i}"].append( Ly )
                        domains[f"p{i}"].append( np.array( [ mx, my ] ) )
        
        else: self.d_coeffs = base_d_coeff

        if any( hard_boundaries ):

            hard_boundaries_geometry = {}
            
            for key, geometries in hard_boundaries.items():

                if key == 'Circle':

                    for i, geometry in enumerate(geometries):


                        hard_boundaries_geometry[f"c{i}"] = [np.array([geometry[1],
                                                                       geometry[2]]),
                                                                       geometry[0]
                                                             ]

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
                        
                        hard_boundaries_geometry[f"p{i}"] = [ np.array([ edge[0] % self.size_x, edge[1] % self.size_y ]) for edge in edges ]
                        
                        hard_boundaries_geometry[f"p{i}"].append( Lx )
                        hard_boundaries_geometry[f"p{i}"].append( Ly )
                        hard_boundaries_geometry[f"p{i}"].append( np.array( [ mx, my ] ) )

            self.hard_boundaries_geometry = hard_boundaries_geometry

        else: self.hard_boundaries_geometries = {}
    
    def base_apply_pbc_vector(self, vec):

        #Apply PBC
        vec[0] = np.where(vec[0] >    self.size_x / 2, vec[0] - self.size_x, vec[0])
        vec[0] = np.where(vec[0] <= - self.size_x / 2, vec[0] + self.size_x, vec[0])

        vec[1] = np.where(vec[1] >    self.size_y / 2, vec[1] - self.size_y, vec[1])
        vec[1] = np.where(vec[1] <= - self.size_y / 2, vec[1] + self.size_y, vec[1])

        return vec


    
