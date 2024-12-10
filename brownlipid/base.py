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
                  grid:                    float = 0.1,
               domains:                     dict = {},
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
        self.nstchk       = nstck
        self.base_d_coeff = base_d_coeff
        self.grid         = grid
        self.output       = output

        #Base checking
        assert self.nsteps >= self.nstxout, "Number of steps must be larger or equal than frequency of output (nstxout)!"
        assert self.nstxout >= self.nstchk, "Frequency of output (nstxout) must be larger than frequency of checkpoints (nstchk)!"

        assert self.nsteps % self.nstxout == 0, "Frequency of output (nstxout) must be multiple of number of steps!"
        assert self.nsteps % self.nstchk  == 0, "Frequency of checkpoints (nstchk) must be multiple of number of steps!"
        assert self.nstxout % self.nstchk == 0, "Frequency of checkpoints (nstchk) must be multiple of frequency of output (nstxout)!"

        #Init domains
        if any( domains ):

            d_coeffs  = np.ones( ( int(self.size_x/self.grid + 1), int(self.size_y/self.grid + 1) ) )
            d_coeffs *= self.base_d_coeff

            grid_coord = np.meshgrid(np.linspace( 0, self.size_x, int(self.size_x/self.grid + 1) ),
                                     np.linspace( 0, self.size_y, int(self.size_y/self.grid + 1) ) 
                                    )

            grid_coord = np.vstack(( grid_coord[0].flatten(), grid_coord[1].flatten() )).T

            for key, geometries in domains.items():

                if key == 'Circle':

                    for geometry in geometries:

                        circle = self.add_circ(r  = geometry[1], mx = geometry[2], my = geometry[3],
                                               dx = self.grid, grid_coord = grid_coord,
                                               size_x = self.size_x, size_y = self.size_y)
                        
                        d_coeffs[circle[:, 0], circle[:, 1]] = geometry[0]

                elif key == 'Rectangle':
                    
                    for geometry in geometries:
        
                        rectangle = self.add_rectangle(mx = geometry[1], my = geometry[2],
                                                       Lx = geometry[3], Ly = geometry[4],
                                                       dx = self.grid, grid_coord = grid_coord,
                                                       size_x = self.size_x, size_y = self.size_y)

                        d_coeffs[rectangle[:, 0], rectangle[:, 1]] = geometry[0]

            self.d_coeffs = d_coeffs

        else: self.d_coeffs = base_d_coeff

    @staticmethod
    def add_circ( mx, my, r, dx, grid_coord, size_x, size_y):

        dmx =  grid_coord[:, 0] - mx
        dmy =  grid_coord[:, 1] - my

        #Apply PBC
        dmx = np.where(dmx >    size_x / 2, dmx - size_x, dmx)
        dmx = np.where(dmx <= - size_x / 2, dmx + size_x, dmx)

        dmy = np.where(dmy >    size_y / 2, dmy - size_y, dmy)
        dmy = np.where(dmy <= - size_y / 2, dmy + size_y, dmy)

        dist_m = np.sqrt( dmx**2 + dmy**2 )

        dom_grid = np.int64( np.round( grid_coord[ dist_m <= r ] / dx ) )

        return dom_grid

    @staticmethod
    def add_rectangle(mx, my, Lx, Ly, dx, grid_coord, size_x, size_y):

        dmx =  grid_coord[:, 0] - mx
        dmy =  grid_coord[:, 1] - my

        #Apply PBC
        dmx = np.where(dmx >   size_x / 2, dmx - size_x, dmx)
        dmx = np.where(dmx <= -size_x / 2, dmx + size_x, dmx)

        dmy = np.where(dmy >   size_y / 2, dmy - size_y, dmy)
        dmy = np.where(dmy <= -size_y / 2, dmy + size_y, dmy)

        cond = (-Lx/2 <= dmx) & (dmx <= Lx/2) & (-Ly/2 <= dmy) & (dmy <= Ly/2)

        dom_grid = np.int64( np.round( grid_coord[ cond ] / dx ) )

        return dom_grid



    
