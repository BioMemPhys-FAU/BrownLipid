# ----PYTHON---- #
from typing import Union, Dict, Any

import numpy as np
from tqdm import tqdm 
import sys

from . import utils
from . import force


class base:

    def __init__(
                           self,
                      input_file:        Union[bool, str] = False,
                          size_x:                    float = 10.0,
                          size_y:                    float = 10.0,
                               N:         Union[int, list] = 1000,
                       comp_name:         Union[str, list] = 'lipid',
                         pbc_dim:                      str = 'xy',
                          nsteps:                      int = 5000,
                              dt:                    float = 1.0,
                         nstxout:                      int = 1,
                          nstchk:                      int = 1,
                    base_d_coeff:       Union[float, list] = 1.,
                 hard_boundaries:                     dict = {},
                        softwall:                     bool = False,
                    bounce_scale:       Union[None, float] = None,
            checkpoint_structure:  Union[None, np.ndarray] = None,
                  random_domains:                     bool = False,
               diffusion_domains:                    float = 0.0,
                 external_forces:                     dict = {},
                            temp:                    float = 298,
               langevin_dynamics:                     dict = {},
               pressure_coupling:        Union[bool, dict] = {},
                          output:                      str = 'output'

                ):

        #Check if input json file is used (Note that if input is given, all other parameters will be overwritten or set to default)
        if input_file:

            assert isinstance(input_file, str), "input_file must be a string!"

            input_params = utils.load_params(input_file)

            #Save input parameters for run info file
            self.run_info_params = input_params.copy()

            #Convert dictionary to list
            input_params = utils.convert_for_json(list(input_params.items()))

            #Definine as variables of class
            for k, v in input_params:
                setattr(self, k, v)

        #No input file: Use base parameters
        else:
            #Saving input parameters for run information file
            run_info_params = locals().copy()
            if 'self' in run_info_params: del run_info_params['self']
            if 'input_file' in run_info_params: del run_info_params['input_file']
            self.run_info_params = utils.convert_for_json(run_info_params)

            self.size_x               = size_x
            self.size_y               = size_y
            self.N                    = N
            self.comp_name            = comp_name
            self.nsteps               = int(nsteps)
            self.dt                   = dt
            self.nstxout              = int(nstxout)
            self.nstchk               = int(nstchk)
            self.checkpoint_structure = checkpoint_structure
            self.base_d_coeff         = base_d_coeff
            self.output               = output
            self.bounce_scale         = bounce_scale
            self.external_forces      = external_forces
            self.temp                 = temp
            self.diffusion_domains    = diffusion_domains
            self.softwall             = softwall
            self.pressure_coupling    = pressure_coupling
            self.langevin_dynamics    = langevin_dynamics

        #Check for multiple components
        if type(self.N) == list or type(self.N) == np.ndarray:
            for param, param_name in zip([self.comp_name, self.base_d_coeff, self.external_forces['sigma'], self.external_forces['epsilon'], self.langevin_dynamics['mass']],
                                         ['comp_name', 'base_d_coeff', 'sigma', 'epsilon', 'mass']):

                assert type(param) == list or type(param) == np.ndarray, f"If N is a list, {param_name} must be a list!"
                assert len(param) == len(self.N) or np.shape(param) == (len(N), len(N)), f"{param_name} must have the same length as N!"

            self.multi_lipid                = True
            self.n_comp                     = len(self.N)
            self.n_lipids_per_comp          = np.array(self.N)
            self.n_lipids_per_comp_cum      = np.cumsum(self.n_lipids_per_comp)
            self.n_lipids_per_comp_cum_0    = np.append(0, self.n_lipids_per_comp_cum)
            self.n_lipids_pairs             = self.n_lipids_per_comp[:, None] * self.n_lipids_per_comp[None,:]
            self.N                          = np.sum(self.N)
            self.comp_name_per_lip          = np.repeat(self.comp_name, self.n_lipids_per_comp)

            self.comp_mask             = np.zeros((self.n_comp, self.N),dtype=bool)
            for i in range(self.n_comp): self.comp_mask[i, self.n_lipids_per_comp_cum_0[i]:self.n_lipids_per_comp_cum_0[i+1]] = True

            self.base_d_coeff               = np.array(self.base_d_coeff)
            self.langevin_dynamics['mass']  = np.array(self.langevin_dynamics['mass'])
            self.external_forces['sigma']   = np.array(self.external_forces['sigma'])
            self.external_forces['epsilon'] = np.array(self.external_forces['epsilon'])

        else:
            self.multi_lipid                = False
            self.comp_name                  = np.array([self.comp_name])
            self.n_comp                     = 1
            self.n_lipids_per_comp          = np.array([self.N])
            self.n_lipids_per_comp_cum      = self.n_lipids_per_comp
            self.n_lipids_pairs             = self.n_lipids_per_comp[:, None] * self.n_lipids_per_comp[None,:]
            self.comp_name_per_lip          = np.repeat(self.comp_name, self.n_lipids_per_comp)
            self.comp_mask                  = np.ones((1, self.N),dtype=bool)

            self.base_d_coeff               = np.array([self.base_d_coeff])
            if any(self.langevin_dynamics): self.langevin_dynamics['mass']  = np.array([self.langevin_dynamics['mass']])

            if any(self.external_forces):
                if type(self.external_forces['sigma']) in [float, int] and type(self.external_forces['epsilon']) in [float, int]:
                    self.external_forces['sigma']   = np.array([[self.external_forces['sigma']]])
                    self.external_forces['epsilon'] = np.array([[self.external_forces['epsilon']]])

        #Precalculation
        self.area                   = self.size_x * self.size_y
        self.RT                     = 8.3145 * 1E-3 * self.temp
        self.NRT                    = self.N * self.RT
        self.NkT                    = self.N * 1.38 * self.temp #1E-23 J
        self.d_coeffs               = np.repeat( self.base_d_coeff, self.n_lipids_per_comp ).reshape(-1,1)

        #Base checking
        assert self.nsteps >= self.nstxout, "Number of steps must be larger or equal than frequency of output (nstxout)!"
        assert self.nstchk >= self.nstxout, "Frequency of output (nstxout) must be smaller than frequency of checkpoints (nstchk)!"

        assert self.nsteps % self.nstxout == 0, "Frequency of output (nstxout) must be multiple of number of steps!"
        assert self.nsteps % self.nstchk  == 0, "Frequency of checkpoints (nstchk) must be multiple of number of steps!"
        assert self.nstchk % self.nstxout == 0, "Frequency of output (nstxout) must be multiple of frequency of checkpoints (nstchk)!"

        self.offsets = np.zeros( (self.N, self.N) , dtype = np.float32)
        
        #----------------------------------------------------------------------------------------------------------------------------------------------
        #PBC Handling
        if input_file: pbc_dim = self.pbc_dim
        pbc_index = []
        pbc_size  = []

        hard_wall_index = []
        hard_wall_size  = []

        for idx, size, dim in zip([0, 1], [self.size_x, self.size_y], ['x', 'y']):

            if dim in pbc_dim:
                if idx == 0: 
                    pbc_index.append(1)
                    pbc_size.append( np.float64(self.size_y) )
                else: 
                    pbc_index.append(0)
                    pbc_size.append( np.float64(self.size_x) )
            else:
                if idx == 0: 
                    hard_wall_index.append(1)
                    hard_wall_size.append( np.float64(self.size_y) )
                else:
                    hard_wall_index.append(0)
                    hard_wall_size.append( np.float64(self.size_x) )
        
        self.pbc_dim   = (pbc_index, pbc_size)
        self.hard_wall = (hard_wall_index, hard_wall_size)

        #----------------------------------------------------------------------------------------------------------------------------------------------
        #Initialize for external forces

        #This parameter is needed later for initializing positions and must be defined also if no external forces are requested
        self.lj_sig = np.array([[-1000]])

        if any(self.external_forces):

            print("Inter-particle forces are requested!")
            print("Setting up LJ...")

            #Pre-calculate lennard-jones parameters to speed up calculations
            self.lj_sig = self.external_forces['sigma']
            self.lj_eps = self.external_forces['epsilon']

            print("Found the following LJ parameters:")
            print("sigma/eps:", self.lj_sig, self.lj_eps )

            #self.rmin           = self.lj_sig * 2**(1/6)

            #Setup interaction matrices
            comp_ids = np.repeat( np.arange(self.n_comp), self.n_lipids_per_comp )

            #Map to sigma values
            self.sigma_matrix = self.lj_sig[comp_ids[:, None], comp_ids[None, :]]

            assert np.shape(self.sigma_matrix) == (self.N, self.N), f'Sigma Matrix has wrong shape! {np.shape(self.sigma_matrix)} != {(self.N, self.N)}'

            #Map to epsilon values
            self.eps_matrix = self.lj_eps[comp_ids[:, None], comp_ids[None, :]]

            assert np.shape(self.eps_matrix) == (self.N, self.N), f'Epsilon Matrix has wrong shape! {np.shape(self.eps_matrix)} != {(self.N, self.N)}'

            self.lj_A12 = 48 * self.eps_matrix * self.sigma_matrix**12
            self.lj_B6  = 24 * self.eps_matrix * self.sigma_matrix**6

            print("Pre-calculated LJ terms")
            print("Lipid/Lipid - A12 - B6")
            print(self.lj_A12[0,0], self.lj_B6[0,0])
            print("Lipid/Domain - A12 - B6")
            print(self.lj_A12[0,-1], self.lj_B6[0,-1])
            print("Domain/Domain - A12 - B6")
            print(self.lj_A12[-1,-1], self.lj_B6[-1,-1])

            #Parameters for neighbour list generation
            self.lj_nstlist = self.external_forces['nstlist']

            #Only squared sums are considered later
            self.lj_cutoff  = self.external_forces['r_vdw']
            self.lj_buffer  = self.external_forces['r_list']

            #Calculate potential at cutoff for potential shift
            sr6_cut = (self.sigma_matrix * self.lj_cutoff)**6
            sr12_cut = sr6_cut**2
            self.pote_shift_matrix = 4 * self.eps_matrix * (sr12_cut - sr6_cut)

            #Precalculate constant part of long-range correction for potential energy
            E_lrc_const_per_pairs = self.n_lipids_pairs * np.pi * self.lj_eps * self.lj_sig**2 * (0.4 * (self.lj_sig / self.lj_cutoff)**10 - (self.lj_sig / self.lj_cutoff)**4)
            self.E_lrc_const = np.sum(E_lrc_const_per_pairs)

            #Prepare array for force_per_particle calculation
            self.force_pp_zeros = np.zeros( (self.N, 2), dtype = np.float32 )

            #Init empty pairlist -> Will be changed in the first step of the simulations
            self.pp_pairlist = np.array([])

            #Prepare a matrix for offsets (hard sphere domains)
            self.offsets = np.zeros( (self.N, self.N) , dtype = np.float32)

            print("System topology")
            print("Total lipids:", self.N)
            print("Lipids per component:", self.n_lipids_per_comp)

        #----------------------------------------------------------------------------------------------------------------------------------------------
        #Initialize for langevin dynamics

        if self.langevin_dynamics != False and isinstance(self.langevin_dynamics, dict):

            if len(self.langevin_dynamics) != 0:

                print("Langevin dynamics are requested!")

                #Check for unknown keys
                unknown = set(self.langevin_dynamics) - {'mass'}
                if unknown: raise KeyError(f"Unknown key(s) in langevin_dynamics: {unknown}")

                #Set given values if present, otherwise set default values
                for k, v in zip(['mass'],
                                np.ones(self.N * 750)):

                    if k in self.langevin_dynamics:
                        setattr(self, k, self.langevin_dynamics[k])
                    else:
                        setattr(self, k, v)

                if any(self.mass < 0): raise ValueError("mass must be positive!")
                if any(self.base_d_coeff <= 0): raise ValueError("base_d_coeff must be positive!")

                #Mass in g/mol
                mass_per_lip = np.zeros(self.N)
                for comp, mask in enumerate(self.comp_mask): mass_per_lip[mask] = self.mass[comp]
                self.mass = mass_per_lip.reshape(-1,1)

                self.gamma = 1.38 * self.temp * 6.022 * 1e3 / (self.mass * self.d_coeffs) # 1/ns

                self.c1 = np.exp(-self.gamma * self.dt)
                self.c3 = np.sqrt((1-self.c1**2) * self.mass * 1.38 * self.temp * 6.022 * 1e-3)  # kg * nm / (ns * mol) = 10^-3 * kJ * ns / (nm * mol)

            # Disable langevin when dict is empty
            else: self.langevin_dynamics = False

        #----------------------------------------------------------------------------------------------------------------------------------------------
        #Initialize for pressure coupling

        if self.pressure_coupling is True:

            print("Using default parameters for pressure coupling!")

            #Set default values
            self.ref_p = 0                          #Reference pressure (tension) in mN/m
            self.compressibility = 5.6e-05            #Isothermal compressibility in m/mN
            self.tau_p = 0.01                      #Rate of pressure adjustment in ns
            self.thresh_p = 1e-21                   #Threshold for coupling to ref_p
            self.nstpcouple = 1                     #Rescaling frequency
            self.K_A = 225                          #Area compressibility
            self.ref_A = self.area               #Reference area

        elif isinstance(self.pressure_coupling, dict) and len(self.pressure_coupling) != 0:

            print("Pressure coupling is enabled!")

            #Check for unknown keys
            unknown = set(self.pressure_coupling) - {'ref_p', 'compressibility', 'tau_p', 'thresh_p', 'nstpcouple', 'K_A', 'ref_A'}
            if unknown: raise KeyError(f"Unknown key(s) in pressure_coupling: {unknown}")

            # Set given values if present, otherwise set default values
            for k, v in zip(['ref_p', 'compressibility', 'tau_p', 'thresh_p', 'nstpcouple', 'K_A', 'ref_A'], [0, 5.6e-05, 0.01, 1e-21, 1, 225, self.area]):

                if k in self.pressure_coupling: setattr(self, k, self.pressure_coupling[k])
                else: setattr(self, k, v)

            if self.ref_p < 0 or self.compressibility < 0 or self.tau_p < 0 or self.thresh_p < 0:
                raise ValueError("Pressure coupling parameters must be positive!")

            assert self.nsteps % self.nstpcouple == 0, "Frequency of pressure coupling (nstpcouple) must be multiple of number of steps!"

            self.nstpcouple = int(self.nstpcouple)

        #Disable pressure coupling when dict is empty or pressure_coupling is set as False
        else: self.pressure_coupling = False

        #Precalculate constant part of long-range correction
        #(Keep in mind that the output virial isn't long-range corrected)
        if self.pressure_coupling != False:

            if any(self.external_forces):
                p_lrc_const_per_pairs = 12 * self.n_lipids_pairs * np.pi * self.lj_eps * self.lj_sig**2 * (0.2 * (self.lj_sig / self.lj_cutoff)**10 - 0.25 * (self.lj_sig / self.lj_cutoff)**4)
                self.p_lrc_const = np.sum(p_lrc_const_per_pairs)

            else : self.p_lrc_const = 0