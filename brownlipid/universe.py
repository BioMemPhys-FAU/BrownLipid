# ----PYTHON---- #
from typing import Union, Dict, Any

import numpy as np
from tqdm import tqdm 

from .base import base

class Universe(base):

    def evolve(self):

        self.w_universe = self.populate_universe_uniform()
        self.u_universe = np.copy( self.w_universe )

        w_storage   = np.zeros( (self.nsteps // self.nstxout, self.N, 2 ), dtype = np.float32 )
        u_storage   = np.zeros( (self.nsteps // self.nstxout, self.N, 2 ), dtype = np.float32 )

        for i in tqdm( range( self.nsteps ) ):
            
            #Check diffusion coefficients
            self.effective_d_coeffs = self.generate_diffusion()

            #Move particles in time
            self.forward_in_time()

            #Apply boundary conditions
            self.apply_pbc()

            if not i % self.nstxout: 
                w_storage[ i // self.nstxout ] = self.w_universe
                u_storage[ i // self.nstxout ] = self.u_universe

        np.save( arr = w_storage, file = "wrap_"   + self.output )
        np.save( arr = u_storage, file = "unwrap_" + self.output )

        self.w_storage = w_storage
        self.u_storage = u_storage

        
    def populate_universe_uniform(self):

        init_pos = np.random.rand( self.N, self.dim )
        
        for i, size in enumerate([self.size_x, self.size_y, self.size_z][:self.dim]):
            init_pos[:, i] *= size

        return init_pos[:, :self.dim]

    def forward_in_time(self):

        factor = np.sqrt( 2 * self.effective_d_coeffs * self.dt ).reshape(-1, 1)

        displace = factor * np.random.randn( self.N, self.dim )

        #Move particles
        self.w_universe += displace
        self.u_universe += displace

    def apply_pbc(self):

        #Apply periodic boundary conditions
        for i, size in enumerate([self.size_x, self.size_y, self.size_z][:self.dim]):
            self.w_universe[:, i] %= size

    def generate_diffusion(self):

        if type( self.d_coeffs ) == float: return np.repeat(self.d_coeffs, self.N)
        
        elif self.d_coeffs.ndim == 2:

            grid_indices = np.int64( np.floor( self.w_universe  / self.grid ) )

            return self.d_coeffs[ grid_indices[:, 0], grid_indices[:, 1] ]
        
        else:
            raise ValueError("Currently I cannot handle the provided diffusion coefficients. Either enter float or 2-dimensional numpy array")

    def mean_square_displacement(self):

        lagtimes = np.arange(1, self.nsteps // self.nstxout)

        msd = np.zeros( lagtimes.shape[0] , dtype = np.float32)
        sd_per_particle = np.zeros( (lagtimes.shape[0], self.N) , dtype = np.float32)

        for i, lag in tqdm(enumerate(lagtimes), total = lagtimes.shape[0]):

            dr = self.u_storage[:-lag, :, :] - self.u_storage[lag:, :, :]
            sqdist = np.square(dr).sum(axis=-1)

            msd[i] = sqdist.mean()
            sd_per_particle[i] = sqdist.mean(axis = 0)

        tau = np.float32(lagtimes)
        tau *= self.dt * self.nstxout

        return tau, msd, sd_per_particle
    
    def mean_square_displacement_distr(self, tau):

        lag = int(np.round(tau / self.dt / self.nstxout))

        dr = self.u_storage[:-lag, :, :] - self.u_storage[lag:, :, :]
        sqdist = np.square(dr).sum(axis=-1)

        sd_per_particle = sqdist.mean(axis = 0)

        print(f'MSD Distribution at {lag * self.dt * self.nstxout / 1000 / 1000} ms') 

        return sd_per_particle
