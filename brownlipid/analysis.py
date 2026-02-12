import numpy as np
import matplotlib.pyplot as plt
import scipy
import tidynamics
from tqdm import tqdm
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

import brownlipid
from .base import base
from brownlipid import pressure
from brownlipid import utils
from brownlipid import force

'''
Analysis of BrownLipid simulations.


'''

class Analysis(base):

    def load_data_unwrap(self, block = None, skip = 1):

        try:
            if self.u_storage.shape[0] == self.nsteps // self.nstxout // skip + 1 and self.u_storage.shape[1] == self.N: return 0
        except: pass

        if type(block) == type(None): block = np.arange( self.N )

        self.u_storage  = np.zeros( (0, len(block), 2), dtype = np.float32 )
        self.time_array = np.zeros( (0)               , dtype = np.float32 )

        for chk_number in range(1, self.nsteps // self.nstchk + 1 ):

            chk_number = str(chk_number)

            if chk_number == "1":

                self.u_storage  = np.vstack( (self.u_storage, np.load(self.output + f"_unwrap.{chk_number.zfill(5)}.npy")[:, block, :] ))
                self.time_array = np.append( self.time_array, np.load(self.output + f"_time.{chk_number.zfill(5)}.npy"))

            else:

                data = np.load(self.output + f"_unwrap.{chk_number.zfill(5)}.npy")
                time = np.load(self.output + f"_time.{chk_number.zfill(5)}.npy")

                self.u_storage  = np.vstack( (self.u_storage, data[:, block, :] ))
                self.time_array = np.append( self.time_array, time  )

        self.u_storage = self.u_storage[::skip, :, :]
        self.time_array = self.time_array[::skip]

        #Validation
        assert self.u_storage.shape[0] == self.nsteps // self.nstxout // skip + 1, "Number of frames is not correct!"

        diff_test = np.diff( self.time_array )

        assert np.allclose( np.diff( self.time_array ), self.dt * self.nstxout * skip, atol = 1E-3, rtol = 0), "Time step distance is not as expected!"
        assert np.allclose( self.dt * self.nstxout * skip, np.diff( self.time_array ), atol = 1E-3, rtol = 0 ), "Time step distance is not as expected!"

        return 1


    def load_data_wrap(self, block = None, skip = 1):

        try:
            if self.w_storage.shape[0] == self.nsteps // self.nstxout // skip + 1: return 0
        except: pass

        if type(block) == type(None): block = np.arange( self.N )

        self.w_storage  = np.zeros( (0, len(block), 2), dtype = np.float32 )
        self.time_array = np.zeros( (0)               , dtype = np.float32 )

        for chk_number in range(1, self.nsteps // self.nstchk + 1 ):

            chk_number = str(chk_number)

            if chk_number == "1":

                self.w_storage  = np.vstack( (self.w_storage, np.load(self.output + f"_wrap.{chk_number.zfill(5)}.npy")[:, block, :] ))
                self.time_array = np.append( self.time_array, np.load(self.output + f"_time.{chk_number.zfill(5)}.npy"))

            else:

                data = np.load(self.output + f"_wrap.{chk_number.zfill(5)}.npy")
                time = np.load(self.output + f"_time.{chk_number.zfill(5)}.npy")

                self.w_storage  = np.vstack( (self.w_storage, data[:, block, :] ))
                self.time_array = np.append( self.time_array, time  )

        self.w_storage = self.w_storage[::skip, :, :]
        self.time_array = self.time_array[::skip]

        print(f"Check {chk_number}: {self.time_array[-1]}")

        #Validation
        assert self.w_storage.shape[0] == ( self.nsteps // self.nstxout // skip + 1), f"Number of frames is not correct! Expected: {self.nsteps // self.nstxout // skip + 1} Got: {self.w_storage.shape[0]}"

        assert np.allclose( np.diff( self.time_array ), self.dt * self.nstxout * skip, rtol = 0, atol = 1E-3), "Time step distance is not as expected!"
        assert np.allclose( self.dt * self.nstxout * skip, np.diff( self.time_array ), rtol = 0, atol = 1E-3 ), "Time step distance is not as expected!"

        return 1


    def load_data_area(self, skip = 1):

        try:
            if self.Area.shape[0] == self.nsteps // self.nstxout // skip + 1: return 0
        except: pass

        self.Area = np.zeros((0), dtype = np.float32)

        for chk_number in range(1, self.nsteps // self.nstchk + 1 ):

            chk_number = str(chk_number)

            if chk_number == "1":

                self.Area = np.append(self.Area, np.load(self.output + f'_Area.{chk_number.zfill(5)}.npy'))

            else:

                data = np.load(self.output + f'_Area.{chk_number.zfill(5)}.npy')

                self.Area = np.append(self.Area, data)

        self.Area = self.Area[::skip]

        #Validation
        assert self.Area.shape[0] == ( self.nsteps // self.nstxout // skip + 1), f"Number of frames is not correct! Expected: {self.nsteps // self.nstxout // skip + 1} Got: {self.Area.shape[0]}"

        return 1


    def load_data_pressure(self, skip = 1):

        try:
            if self.Pressure.shape[0] == self.nsteps // self.nstxout // skip + 1: return 0
        except: pass

        self.Pressure = np.zeros((0), dtype = np.float32)

        for chk_number in range(1, self.nsteps // self.nstchk + 1 ):

            chk_number = str(chk_number)

            if chk_number == "1":

                self.Pressure = np.append(self.Pressure, np.load(self.output + f'_Pressure.{chk_number.zfill(5)}.npy'))

            else:

                data = np.load(self.output + f'_Pressure.{chk_number.zfill(5)}.npy')

                self.Pressure = np.append(self.Pressure, data)

        self.Pressure = self.Pressure[::skip]

        #Validation
        assert self.Pressure.shape[0] == ( self.nsteps // self.nstxout // skip + 1), f"Number of frames is not correct! Expected: {self.nsteps // self.nstxout // skip + 1} Got: {self.Pressure.shape[0]}"

        return 1

    def load_data_velocity(self, block=None, skip=1):

        try:
            if self.Velocity.shape[0] == self.nsteps // self.nstxout // skip + 1: return 0
        except:
            pass

        if type(block) == type(None): block = np.arange(self.N)

        self.Velocity = np.zeros((0, len(block), 2), dtype=np.float32)
        self.time_array = np.zeros((0), dtype=np.float32)

        for chk_number in range(1, self.nsteps // self.nstchk + 1):

            chk_number = str(chk_number)

            if chk_number == "1":

                self.Velocity = np.vstack(
                    (self.Velocity, np.load(self.output + f"_Velocity.{chk_number.zfill(5)}.npy")[:, block, :]))
                self.time_array = np.append(self.time_array, np.load(self.output + f"_time.{chk_number.zfill(5)}.npy"))

            else:

                data = np.load(self.output + f"_Velocity.{chk_number.zfill(5)}.npy")
                time = np.load(self.output + f"_time.{chk_number.zfill(5)}.npy")

                self.Velocity = np.vstack((self.Velocity, data[:, block, :]))
                self.time_array = np.append(self.time_array, time)

        self.Velocity = self.Velocity[::skip, :, :]
        self.time_array = self.time_array[::skip]

        print(f"Check {chk_number}: {self.time_array[-1]}")

        # Validation
        assert self.Velocity.shape[0] == (
                    self.nsteps // self.nstxout // skip + 1), f"Number of frames is not correct! Expected: {self.nsteps // self.nstxout // skip + 1} Got: {self.Velocity.shape[0]}"

        assert np.allclose(np.diff(self.time_array), self.dt * self.nstxout * skip, rtol=0,
                           atol=1E-3), "Time step distance is not as expected!"
        assert np.allclose(self.dt * self.nstxout * skip, np.diff(self.time_array), rtol=0,
                           atol=1E-3), "Time step distance is not as expected!"

        return 1


    def mean_square_displacement(self, begin = 0, stop = None, skip = None, fft = True, block = None):

        """
        Mean Square Displacement

        Calculate the Mean Square Displacement of selected particles for different lag times.

        Parameters
        ----------

        begin := float
            Start time for analysis (ns)
        stop := float
            Stop time for analysis (ns)
        skip := float
            Time to skip between frames for analysis (ns)
        block := array
            Indices of particles to include in the analysis. If None, all particles are considered
        fft := bool
            Use Fourier Transforms

        """

        if stop == None: stop = self.nsteps * self.dt
        if skip == None: skip = self.dt * self.nstxout

        assert stop <= self.nsteps * self.dt , f'Error. There are only {self.nsteps * self.dt} ns simulation time!'
        assert begin <= self.nsteps * self.dt, f'Error. There are only {self.nsteps * self.dt} ns simulation time!'
        assert np.round((stop - begin) % skip) == 0, 'Number of frames is not divisible by skip!'

        if type(block) == type(None): block = np.arange( self.N )

        #Convert time to frames
        skip  = int( np.round( skip  / self.dt / self.nstxout) )
        begin = int( np.round( begin / self.dt / self.nstxout ) // skip)
        stop  = int( np.round( stop  / self.dt / self.nstxout ) // skip) + 1

        #Number of steps for analysis
        nsteps_analysis = (stop - begin) // skip

        print(f"Calculating MSD...")
        print(f"Start: Frame {begin}")
        print(f"Stop : Frame {stop}")
        print("")

        #Load data
        self.load_data_unwrap(skip = skip, block = block)

        u_storage_analysis = np.copy( self.u_storage[begin:stop, :, :] )

        assert nsteps_analysis == u_storage_analysis.shape[0], 'Not correct number of frames'

        lagtimes = np.arange(0, nsteps_analysis, skip)

        print("Start analysis...")

        #if fft == True: msd, sd_per_particle = utils.MSD_fft_ax(pos = u_storage_analysis.astype(np.float64))

        if fft == True:

            sd_per_particle = np.zeros((self.N, len(lagtimes)))
            for n in range(self.N):

                sd_per_particle[n] = tidynamics.msd(pos = u_storage_analysis[:, n, :].astype(np.float64))

            msd = np.mean(sd_per_particle, axis = 0)

        else: msd, sd_per_particle = utils.evaluate_lagtimes(pos = u_storage_analysis, lagtimes = lagtimes, N = u_storage_analysis.shape[1] )

        #Convert lagtimes array to physical time
        tau = lagtimes.astype( np.float32 )
        tau = tau * self.dt * self.nstxout

        return tau, msd, sd_per_particle

    def get_rdf(self, exp_density, begin = 0, stop = None, skip = None, block = None, r_max = 5, binwidth = 0.5, bulk = False):

        """
        Radial Distribution Function

        Calculates the radial distribution function (RDF) and the cumulative distribution function (CDF) for the selected particles.

        Parameters
        ----------
        exp_density : float
            Density of the system (1/nm^2)
        begin := float
            Start time for analysis (ns)
        stop := float
            Stop time for analysis (ns)
        skip := float
            Time to skip between frames for analysis (ns)
        block : array_like, optional
            Indices of particles to include in the analysis. If None, all particles are considered
        r_max : float
            Maximum radius for RDF calculation (ns)
        binwidth : float
            Width of the bins for the RDF histogram
        bulk : bool
            If True, calculates the RDF for a bulk system, otherwise for a self-RDF

        """

        if type(stop) == type(None): stop = self.nsteps * self.dt
        if type(skip) == type(None): skip = self.dt * self.nstxout

        assert stop <= self.nsteps * self.dt , f'Error. There are only {self.nsteps * self.dt} ns simulation time!'
        assert begin <= self.nsteps * self.dt, f'Error. There are only {self.nsteps * self.dt} ns simulation time!'
        assert np.round((stop - begin) % skip) == 0, 'Number of frames is not divisible by skip!'

        if block == None: block = np.arange( self.N )

        #Convert time to frames
        skip  = int( np.round( skip  / self.dt / self.nstxout) )
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
        self.load_data_wrap(skip = skip, block=block)

        w_storage_analysis = np.copy( self.w_storage[begin:stop, :, :] )

        self.load_data_area(skip = skip)
        area_analysis = np.copy( self.Area[begin:stop] )

        assert nsteps_analysis == w_storage_analysis.shape[0], 'Not correct number of frames'

        if bulk: binmids, rdf, cdf, pdf = utils.self_bulk_rdf(exp_density = exp_density, pos = w_storage_analysis, pbc_dim = self.pbc_dim, r_max = r_max, binwidth = binwidth, area = area_analysis, domain_coords = self.domain_coords, radii = self.domain_radii, rmin = self.rmin)
        else:    binmids, rdf, cdf, pdf = utils.self_rdf(exp_density = exp_density, pos = w_storage_analysis, area = area_analysis, pbc_dim = self.pbc_dim, r_max = r_max, binwidth = binwidth)

        return binmids, rdf.mean(0), cdf.mean(0), pdf.mean(0)

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

        dr = u_storage_analysis[:-lag, :, :] - self.u_storage[lag:, :, :]
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


        #------------------------------------------------------------
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

    def export_trajectory(self, atomname = 'B1', resname = 'LIP', output = None, pml_file = False, begin=0, stop=None, skip=1, block = None):

        """
        Export trajectory to gromacs format


        Parameters
        ----------

        atomname := str
            Atom name to be written to the output .gro file
        resname := str
            Residue name to be written to the output .gro file
        output := str
            Name for the output .gro and .xtc files which gets appended to the standard output (self.output)
        pml_file := bool or str
            Generate a PyMOL script to load the trajectory automatically with bead radius = sigma / 2
        begin := float
            Start time for export (ns)
        stop := float
            Stop time for export (ns)
        skip := float
            Time to skip between frames for analysis (ns)
        block := array
            Indices of particles to include in the analysis. If None, all particles are considered

        """

        if type(output) == type(None): output = self.output + '_trajectory'
        else:
            assert type(output) == str, 'output must be a string!'
            output = self.output + '_' + output

        if type(stop) == type(None): stop = self.nsteps * self.dt
        if type(skip) == type(None): skip = self.dt * self.nstxout

        assert stop <= self.nsteps * self.dt , f'Error. There are only {self.nsteps * self.dt} ns simulation time!'
        assert begin <= self.nsteps * self.dt, f'Error. There are only {self.nsteps * self.dt} ns simulation time!'
        assert np.round((stop - begin) % skip) == 0, 'Number of frames is not divisible by skip!'
        assert type(pml_file) in [bool, str], 'pml_file must be a boolean or a string!'

        #Convert time to frames
        skip  = int( np.round( skip  / self.dt / self.nstxout) )
        begin = int( np.round( begin / self.dt / self.nstxout ) // skip)
        stop  = int( np.round( stop  / self.dt / self.nstxout ) // skip) + 1

        if type(block) == type(None): block = np.arange( self.N )
        N_analysis = len(block)

        #Number of steps for analysis
        nsteps_analysis = (stop - begin)

        print(f"Loading data...")
        print(f"Start: Frame {begin}")
        print(f"Stop : Frame {stop}")
        print("")

        #Load data
        self.load_data_wrap(skip = skip, block = block)

        w_storage_analysis = np.copy( self.w_storage[begin:stop, :, :] )

        assert nsteps_analysis == w_storage_analysis.shape[0], 'Not correct number of frames'

        traj = np.zeros((nsteps_analysis, N_analysis, 3), dtype = np.float32)
        traj[:,:,0:2] = w_storage_analysis * 10 #MDA uses Angström

        self.load_data_area(skip = skip)

        area_analysis = np.copy( self.Area[begin:stop] )
        L = np.sqrt(area_analysis) * 10 #MDA uses Angström

        #Define box dimension with format: [Lx, Ly, Lz, alpha, beta, gamma]
        dim = np.column_stack((L, L, np.ones(nsteps_analysis), np.full(nsteps_analysis, 90), np.full(nsteps_analysis, 90), np.full(nsteps_analysis, 90)))

        #Create empty MDA universe
        u = mda.Universe.empty(n_atoms=N_analysis, n_residues=N_analysis, atom_resindex=block, trajectory=True)

        #Add topology
        u.add_TopologyAttr('resname', [resname] * N_analysis)
        u.add_TopologyAttr('name', [atomname] * N_analysis)

        #Load trajectory
        u.load_new(traj, order='fac', dimensions=dim)

        print(f"Starting export...\n")

        #Write .gro and .xtc files

        #Save first frame
        u.trajectory[0]
        u.atoms.write(f'{output}.gro')

        #Write .xtc trajectory
        with mda.Writer(f'{output}.xtc', n_atoms=N_analysis) as W:
            for ts in u.trajectory:
                W.write(u)

        print(f'Export finished!\nCreated {output}.gro and {output}.xtc')

        if pml_file:

            #Use sigma as distance => sigma/2 as radius of bead
            vdw_r = self.lj_sig / 2 * 10 #MDA uses Angström

            pml_content = f"""
            
            load {output}.gro
            load {output}.xtc
            
            show cell
            
            alter all, vdw={vdw_r:.4g}

            """

            if type(pml_file) == type(True): output = self.output + '_trajectory_script'
            else: output = self.output + pml_file

            with open(f'{output}.pml', 'w') as f:
                f.write(pml_content)

            print(f'Created {output}.pml')


    def get_pair_distances(self, begin=0, stop=None, skip=None, block = None):

        """
        Calculates the pair distances for the specified particles within the given time range



        Parameters
        ----------

        begin := float
            Start time for export (ns)
        stop := float
            Stop time for export (ns)
        skip := float
            Time to skip between frames for analysis (ns)
        block := array
            Indices of particles to include in the analysis. If None, all particles are considered

        Returns
        -------
        pair_distances := np.ndarray
            Flat array of all pair distances calculated.

        """

        if type(stop) == type(None): stop = self.nsteps * self.dt
        if type(skip) == type(None): skip = self.dt * self.nstxout

        assert stop <= self.nsteps * self.dt , f'Error. There are only {self.nsteps * self.dt} ns simulation time!'
        assert begin <= self.nsteps * self.dt, f'Error. There are only {self.nsteps * self.dt} ns simulation time!'
        assert np.round((stop - begin) % skip) == 0, 'Number of frames is not divisible by skip!'

        #Convert time to frames
        skip  = int( np.round( skip  / self.dt / self.nstxout) )
        begin = int( np.round( begin / self.dt / self.nstxout ) // skip)
        stop  = int( np.round( stop  / self.dt / self.nstxout ) // skip) + 1

        if type(block) == type(None): block = np.arange( self.N )
        N_analysis = len(block)

        #Number of steps for analysis
        nsteps_analysis = (stop - begin)

        print(f"Calculating pair distances...")
        print(f"Start: Frame {begin}")
        print(f"Stop : Frame {stop}")
        print("")

        self.load_data_wrap(skip = skip, block = block)
        w_storage_analysis = np.copy(self.w_storage[begin:stop, :, :])

        self.load_data_area(skip = skip)
        area_analysis = np.copy( self.Area[begin:stop] )
        L = np.sqrt(area_analysis)

        pbc_dim_analysis = np.array([self.pbc_dim[0], np.zeros(2)])

        distances = []

        for i in tqdm(range(nsteps_analysis)):

            pbc_dim_analysis[1,:] = L[i]

            dist_mat, _ = utils.distance_matrix_NxN(
                pos = w_storage_analysis[i, :, :],
                N = N_analysis,
                pbc_dim = pbc_dim_analysis,
                offsets = self.offsets
            )

            #Append only valid distances (not inf)
            distances.append(dist_mat[np.isfinite(dist_mat)])

        self.pair_distances = np.concatenate(distances)

        return self.pair_distances