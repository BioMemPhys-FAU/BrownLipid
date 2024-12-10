# ----PYTHON---- #
from typing import Union, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm 

from .base import base

class Universe(base):

    def evolve(self):

        self.w_universe = self.populate_universe_uniform()
        self.u_universe = np.copy( self.w_universe )
        
        w_storage = np.zeros( (self.nstchk // self.nstxout, self.N, self.dim + 1 ), dtype = np.float32 )
        u_storage = np.zeros( (self.nstchk // self.nstxout, self.N, self.dim + 1 ), dtype = np.float32 )

        for i in tqdm( range( self.nsteps ) ):
            
            #Check diffusion coefficients
            self.effective_d_coeffs = self.generate_diffusion()

            #Move particles in time
            self.forward_in_time()

            #Apply boundary conditions
            self.apply_pbc()

            #Write current state of the universe into storage arrays
            if not i % self.nstxout: 

                #Calculate current time
                time = self.dt * i

                w_storage[ i // self.nstxout, :, 0  ] = time
                u_storage[ i // self.nstxout, :, 0  ] = time

                w_storage[ i // self.nstxout, :, 1: ] = self.w_universe
                u_storage[ i // self.nstxout, :, 1: ] = self.u_universe

                #If check pointing is requested then write current storage arrays to disk
                #and renew them
                if not i % self.nstchk and i > 0:

                    #Number of current check point
                    chk_number = str(self.i // self.nstchk)
                    
                    np.save( arr = w_storage, file = self.output +   f"_wrap.{chk_number.zfill(5)}" )
                    np.save( arr = u_storage, file = self.output + f"_unwrap.{chk_number.zfill(5)}" )
        
                    w_storage = np.zeros( (self.nstchk // self.nstxout, self.N, self.dim + 1 ), dtype = np.float32 )
                    u_storage = np.zeros( (self.nstchk // self.nstxout, self.N, self.dim + 1 ), dtype = np.float32 )

                    
        
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

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Analysis part

    def load_data_unwrap(self):

        try:
            if self.u_storage.shape[0] == self.nsteps // self.nstxout: return 0
        except: pass
        
        self.u_storage = np.zeros( (0, self.N, self.dim + 1 ), dtype = np.float32 )

        for chk_number in range(1, self.nsteps // self.nstchk + 1 ):

            chk_number = str(chk_number)

            self.u_storage = np.vstack( (u_storage, np.load(self.output + f"_unwrap.{chk_number.zfill(5)}") ))

        #Validation
        assert self.u_storage.shape[0] == self.nsteps // self.nstxout, "Number of frames is not correct!"

        assert np.allclose(np.diff(self.u_storage[:, 0]), self.nstxout * self.dt), "Validation of loaded unwrapped trajectory was not successful!"
        assert np.allclose(self.nstxout * self.dt, np.diff(self.u_storage[:, 0])), "Validation of loaded unwrapped trajectory was not successful!"

        return 1
    
    def load_data_wrap(self):

        try:
            if self.w_storage.shape[0] == self.nsteps // self.nstxout: return 0
        except: pass
        
        self.w_storage = np.zeros( (0, self.N, self.dim + 1 ), dtype = np.float32 )

        for chk_number in range(1, self.nsteps // self.nstchk + 1 ):

            chk_number = str(chk_number)

            self.w_storage = np.vstack( (w_storage, np.load(self.output + f"_wrap.{chk_number.zfill(5)}") ))

        #Validation
        assert self.w_storage.shape[0] == self.nsteps // self.nstxout, "Number of frames is not correct!"

        assert np.allclose(np.diff(self.w_storage[:, 0]), self.nstxout * self.dt), "Validation of loaded wrapped trajectory was not successful!"
        assert np.allclose(self.nstxout * self.dt, np.diff(self.w_storage[:, 0])), "Validation of loaded wrapped trajectory was not successful!"

        return 1


    def mean_square_displacement(self, skip):

        print(f"Calculate MSD from Frame 1/tau {1*self.dt * self.nstxout}ns to Frame {self.nsteps // self.nstxout}/tau {(self.nsteps // self.nstxout) * self.dt * self.nstxout} with skip Frame {skip/(self.dt* self.nstxout)}/tau {skip}ns")

        #Load data
        self.load_data_unwrap()

        skip /= (self.dt * self.nstxout)

        lagtimes = np.arange(1, self.nsteps // self.nstxout, int(skip))

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
       
        #Load data
        self.load_data_unwrap()

        lag = int(np.round(tau / (self.dt * self.nstxout) ))

        dr = self.u_storage[:-lag, :, :] - self.u_storage[lag:, :, :]
        sqdist = np.square(dr).sum(axis=-1)

        sd_per_particle = sqdist.mean(axis = 0)

        print(f'MSD Distribution at {lag * self.dt * self.nstxout / 1000 / 1000} ms') 

        return sd_per_particle
    
    @staticmethod
    def mean_square_displacement_fit(tau, msd):
        
        #tau -> ns
        #msd -> nm2
        DiffCoeff, Intercept = np.polyfit(x = tau, y = msd, deg = 1)

        #DiffCoeff -> nm2/ns -> um2/ms
        #Intercept -> nm2

        return DiffCoeff, Intercept

    @staticmethod
    def plot_log_histogram_mean_square_displacement_distr(sd_per_particle, lo_limit = 1E-4, up_limit = 1.0, nbins = 51, color = 'red'):

        """
        Plot histogram with log-space bins

        Parameters
        ----------

        sd_per_particle := numpy.ndarray
            Squared-Displacement per particle at a single time-lag tau (expected unit: square-micrometer)
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
                     color = color
                    )

        #Scale
        plt.xscale('log')

        #Label
        plt.ylabel('Number of Trajectories')
        plt.xlabel(r'$\mu$m$^2$')

        


