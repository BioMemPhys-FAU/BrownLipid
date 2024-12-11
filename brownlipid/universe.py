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
        
        w_storage  = np.zeros( (self.nstchk // self.nstxout, self.N, self.dim ), dtype = np.float32 )
        u_storage  = np.zeros( (self.nstchk // self.nstxout, self.N, self.dim ), dtype = np.float32 )

        time_array = np.zeros(  self.nstchk // self.nstxout, dtype = np.float32 )
        
        #Store initial frame
        w_storage[0] = self.w_universe
        u_storage[0] = self.u_universe

        time_array[0] = 0

        idx_norm = self.nstchk // self.nstxout

        for i in tqdm(range(1, self.nsteps + 1) ):
            
            #Check diffusion coefficients
            self.effective_d_coeffs = self.generate_diffusion()

            #Move particles in time
            self.forward_in_time()
            
            #Apply boundary conditions
            self.apply_pbc()

            self.hard_boundaries()

            #Apply boundary conditions
            self.apply_pbc()

            #Write current state of the universe into storage arrays
            if not i % self.nstxout: 

                #Calculate current time
                time = self.dt * i

                if i == self.nsteps:

                    w_storage = np.vstack((w_storage, self.w_universe.reshape(1, self.N, self.dim)))
                    u_storage = np.vstack((u_storage, self.u_universe.reshape(1, self.N, self.dim)))
                    
                    time_array = np.append(time_array, time)
                
                #If check pointing is requested then write current storage arrays to disk
                #and renew them
                if not i % self.nstchk:
                
                    #Number of current check point
                    chk_number = str(i // self.nstchk)

                    np.save( arr = time_array, file = self.output +   f"_time.{chk_number.zfill(5)}" )
                    
                    np.save( arr = w_storage,  file = self.output +   f"_wrap.{chk_number.zfill(5)}" )
                    np.save( arr = u_storage,  file = self.output + f"_unwrap.{chk_number.zfill(5)}" )
        
                    w_storage = np.zeros( (self.nstchk // self.nstxout, self.N, self.dim ), dtype = np.float32 )
                    u_storage = np.zeros( (self.nstchk // self.nstxout, self.N, self.dim ), dtype = np.float32 )
        
                    time_array = np.zeros(  self.nstchk // self.nstxout, dtype = np.float32 )

                #--------------------------------------------------------------------------------------------------
                
                w_storage[ (i // self.nstxout) % idx_norm, :, : ] = self.w_universe
                u_storage[ (i // self.nstxout) % idx_norm, :, : ] = self.u_universe
                
                time_array[ (i // self.nstxout) % idx_norm ] = time



                    
        
    def populate_universe_uniform(self):

        init_pos = np.random.rand( self.N, self.dim )
        
        for i, size in enumerate([self.size_x, self.size_y, self.size_z][:self.dim]):
            init_pos[:, i] *= size
        
        #Resample points if they are in hard boundaries
        if not any(self.hard_boundaries_geometry): pass

        else:
        
            print('Found applied hard boundaries!')
            print('Start resampling...')

            grid_indices = np.int64( np.floor( init_pos  / self.grid ) )

            geom_indices = self.hard_boundaries_grid[ grid_indices[:, 0], grid_indices[:, 1] ]

            while any(geom_indices[ geom_indices != '']):

                init_pos[ geom_indices != '' ] = np.random.rand( np.sum(geom_indices != ''), self.dim )

                for i, size in enumerate([self.size_x, self.size_y, self.size_z][:self.dim]):
                
                    init_pos[ geom_indices != '' ][:, i] *= size
            
                grid_indices = np.int64( np.floor( init_pos  / self.grid ) )

                geom_indices = self.hard_boundaries_grid[ grid_indices[:, 0], grid_indices[:, 1] ]

            print('Resampling finished!')

        assert init_pos[:, 0].max() <= self.size_x 
        assert init_pos[:, 1].max() <= self.size_y
        if self.dim == 3: assert init_pos[:, 2].max() <= self.size_z 
        
        assert init_pos[:, 0].min() >= 0
        assert init_pos[:, 1].min() >= 0
        if self.dim == 3: assert init_pos[:, 2].min() >= 0

        return init_pos[:, :self.dim]

    def forward_in_time(self):

        factor = np.sqrt( 2 * self.effective_d_coeffs * self.dt ).reshape(-1, 1)

        self.displace = factor * np.random.randn( self.N, self.dim )

        #Move particles
        self.w_universe += self.displace
        self.u_universe += self.displace

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

    @staticmethod
    def calc_reflection(d, n):

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

        return d - 2 * np.sum(d * n, axis = 1).reshape(-1, 1) * n

    def get_normals_circle(self, points, mid):

        """
        Calculate the normal vector of a two dimensional circle for a point

        points := numpy.ndarray
            coordinates of reflected points
        index := numpy.ndarray
            index of polygons
        
        """

        norm = points - mid

        #Apply PBC
        norm[:, 0] = np.where(norm[:, 0] >    self.size_x / 2, norm[:, 0] - self.size_x, norm[:, 0])
        norm[:, 0] = np.where(norm[:, 0] <= - self.size_x / 2, norm[:, 0] + self.size_x, norm[:, 0])

        norm[:, 1] = np.where(norm[:, 1] >    self.size_y / 2, norm[:, 1] - self.size_y, norm[:, 1])
        norm[:, 1] = np.where(norm[:, 1] <= - self.size_y / 2, norm[:, 1] + self.size_y, norm[:, 1])


        norm /= np.linalg.norm( norm, axis = 1).reshape(-1, 1)

        return norm

    @staticmethod
    def point_on_line(points, polygon_point_a, polygon_point_b, size_x, size_y):

        ap = points - polygon_point_a
        bp = polygon_point_a - polygon_point_b

        l = polygon_point_a + (np.dot( ap,bp ) / np.dot(bp,bp)).reshape(-1,1) * bp
        
        pl = l - points

        #Apply PBC
        pl[:, 0] = np.where(pl[:, 0] >    size_x / 2, pl[:, 0] - size_x, pl[:, 0])
        pl[:, 0] = np.where(pl[:, 0] <= - size_x / 2, pl[:, 0] + size_x, pl[:, 0])

        pl[:, 1] = np.where(pl[:, 1] >    size_y / 2, pl[:, 1] - size_y, pl[:, 1])
        pl[:, 1] = np.where(pl[:, 1] <= - size_y / 2, pl[:, 1] + size_y, pl[:, 1])

        dist_pl = np.linalg.norm(pl, axis = 1)

        return pl, dist_pl
    
    def get_normals_polyon(self, points, edges):

        """
        Get the normal vector of a two dimensional two dimension polygon for a point

        points := numpy.ndarray
            coordinates of reflected points
        edges := list
            list of polygon edges
        
        """

        number_of_vertices = len(edges) - 1

        point2lines      = np.zeros( (number_of_vertices, points.shape[0], 2), dtype = np.float32)
        dist_point2lines = np.zeros( (number_of_vertices, points.shape[0]),    dtype = np.float32)
        
        for i in range( len(edges) - 2 + 1 ):

            polygon_point_a = edges[i][:, 0]
            polygon_point_b = edges[i+1][:, 0]

            point2lines[i], dist_point2lines[i] = self.point_on_line(points = points, 
                                                                polygon_point_a = polygon_point_a,
                                                                polygon_point_b = polygon_point_b,
                                                                size_x = self.size_x,
                                                                size_y = self.size_y)

        arg_mindist_point2lines = np.argmin(dist_point2lines, axis = 0)

        min_point2lines = point2lines[arg_mindist_point2lines, :, :]

        norm = min_point2lines / dist_point2lines[arg_mindist_point2lines].reshape(-1, 1)

        return norm[0]

    
    def hard_boundaries(self):
        
        if not any(self.hard_boundaries_geometry): pass

        else:
        
            grid_indices = np.int64( np.floor( self.w_universe  / self.grid ) )

            geom_indices = self.hard_boundaries_grid[ grid_indices[:, 0], grid_indices[:, 1] ]

            for geom_index in np.unique(geom_indices):

                if geom_index == '': continue

                #Circular hard boundary
                if 'c' in geom_index:
        
                    mid = self.hard_boundaries_geometry[geom_index]
                
                    norm = self.get_normals_circle(points = self.w_universe[geom_indices == geom_index],
                                                   mid    = mid 
                                                  )
                    
                    reflected = self.calc_reflection(d = self.displace[geom_indices == geom_index],
                                                     n = norm 
                                                    )

                    #----------------------------------------------------------------------------------
                    #Testing
                    #TODO Make it applicable for PBC
                    p_prev = self.w_universe[geom_indices == geom_index] - self.displace[geom_indices == geom_index]

                    pos_p_prev = p_prev    - self.w_universe[geom_indices == geom_index]
                    pos_ref    = reflected# - self.w_universe[geom_indices == geom_index]

                    dot_pp = np.sum(pos_p_prev * norm, axis = 1)
                    dot_pr = np.sum(pos_ref    * norm, axis = 1)

                    dot_pp /= np.linalg.norm(pos_p_prev, axis = 1)
                    dot_pr /= np.linalg.norm(pos_ref   , axis = 1)

                    assert np.allclose(dot_pr, dot_pp), "In angle is not equal out angle"
                    assert np.allclose(dot_pp, dot_pr), "In angle is not equal out angle"
                    #----------------------------------------------------------------------------------

                    self.w_universe[geom_indices == geom_index] += reflected

                #Square hard boundary
                if 'p' in geom_index:
                    
                    edges = self.hard_boundaries_geometry[geom_index]
    
                    norm = self.get_normals_polyon(points = self.w_universe[geom_indices == geom_index],
                                                   edges  = edges
                                                   )
                
                    reflected = self.calc_reflection(d = self.displace[geom_indices == geom_index],
                                                     n = norm 
                                                     )
                    
                    self.w_universe[geom_indices == geom_index] += reflected


            
            #reflection_vector = self.calc_reflection( d = displace_in_domains, n = )







    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Analysis part

    def load_data_unwrap(self):

        try:
            if self.u_storage.shape[0] == self.nsteps // self.nstxout: return 0
        except: pass
        
        self.u_storage  = np.zeros( (0, self.N, self.dim ), dtype = np.float32 )
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
        
        self.w_storage  = np.zeros( (0, self.N, self.dim ), dtype = np.float32 )
        self.time_array = np.zeros( (0)                   , dtype = np.float32 )

        for chk_number in range(1, self.nsteps // self.nstchk + 1 ):

            chk_number = str(chk_number)

            self.w_storage  = np.vstack( (self.w_storage, np.load(self.output + f"_wrap.{chk_number.zfill(5)}.npy") ))

            self.time_array = np.append( self.time_array, np.load(self.output +   f"_time.{chk_number.zfill(5)}.npy")  ) 

        #Validation
        assert self.w_storage.shape[0] == (self.nsteps // self.nstxout) + 1, "Number of frames is not correct!"

        assert np.allclose( np.diff( self.time_array ), self.dt * self.nstxout ), "Time step distance is not as expected!"
        assert np.allclose( self.dt * self.nstxout, np.diff( self.time_array ) ), "Time step distance is not as expected!"

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

        


