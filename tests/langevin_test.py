import brownlipid
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pytest

from brownlipid import pressure
from brownlipid import utils
from brownlipid import force

def test_generate_initial_momenta():

    N = int(4e4)
    mass = 1
    mass_per_lip = (np.ones(N) * mass).reshape(-1, 1)
    comp_mask = np.zeros(N)

    temp = 310
    R = 8.31446

    for _ in range(20):
        momentum = utils.generate_initial_momenta(N, mass_per_lip, temp)

        assert np.isclose(momentum.sum(), 0, atol=1E-3), f'Momentum sum is not zero: {momentum.sum()}'

        v_sq = np.sum((momentum * 1e3 / mass)**2)
        T = ((mass * 1e-3) * v_sq) / (R * (2 * N - 2))

        assert np.isclose(T, temp, atol=5), f'Temperature is not correct: {T}'

    sigma_p = np.sqrt(mass * 1e-3 * R * temp)

    plt.figure(figsize=(10,5))
    _, bins, _ = plt.hist(momentum[:,0], bins = 100, density=True, alpha=0.7, label = 'x' )

    theoretical = (1 / (sigma_p * np.sqrt(2 * np.pi))) * np.exp(-(bins)**2 / (2 * sigma_p**2))

    plt.hist(momentum[:,1], bins = 100, density=True, alpha=0.6, label = 'y' )
    plt.plot(bins, theoretical, label = 'Theoretical' )
    plt.tight_layout()
    plt.show()



def test_generate_initial_momenta_multi_comp():

    N = [50000, 20000, 200000]

    n_comp                     = len(N)
    n_lipids_per_comp          = np.array(N)
    n_lipids_per_comp_cum      = np.cumsum(n_lipids_per_comp)
    n_lipids_per_comp_cum_0    = np.append(0, n_lipids_per_comp_cum)
    N                          = np.sum(N)

    comp_mask             = np.zeros((n_comp, N),dtype=bool)
    for i in range(n_comp): comp_mask[i, n_lipids_per_comp_cum_0[i]:n_lipids_per_comp_cum_0[i+1]] = True

    mass = [4,5,2]
    mass_per_lip = np.zeros(N)
    for comp, mask in enumerate(comp_mask): mass_per_lip[mask] = mass[comp]

    temp = 310
    R = 8.31446

    T_exp_var = np.sqrt(1/N)

    T_array = []

    for _ in range(50):

        mass_per_lip = mass_per_lip.reshape(-1, 1)

        momentum = utils.generate_initial_momenta(N, mass_per_lip, temp)

        assert np.isclose(momentum.sum(), 0, atol=1E-3), f'Momentum sum is not zero: {momentum.sum()}'

        v = (momentum * 1e3) / mass_per_lip

        double_E_kin = np.sum((mass_per_lip * 1e-3) * v**2)

        T = double_E_kin / (R * (2 * N - 2))

        T_array.append(T)

    assert np.isclose(np.mean(T_array), temp, rtol=T_exp_var), f'Temperature is not correct: np.mean(T_array), expected: {temp} +- {temp * T_exp_var}'
    print(f'Temperature: {np.mean(T_array)}, expected: {temp} +- {temp * T_exp_var}')

    plt.figure(figsize=(12, 8))

    colors = ['blue', 'red', 'green']

    for i, m in enumerate(mass):

        mask = comp_mask[i]

        moms_subset = momentum[mask, 0] #only x

        sigma_p_i = np.sqrt(m * 1e-3 * R * temp)

        _, bins, _ = plt.hist(moms_subset, bins=50, density=True,
                              alpha=0.7, color=colors[i%len(colors)], label=f'Mass: {m} g/mol')

        theoretical = (1 / (sigma_p_i * np.sqrt(2 * np.pi))) * np.exp(-(bins)**2 / (2 * sigma_p_i**2))
        plt.plot(bins, theoretical, color=colors[i%len(colors)], linestyle='--', label=f'Theoretical')

    plt.title("Momentum distribution per component")
    plt.xlabel("Momentum (kg * nm / (ns * mol))")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def test_velocity():
    #DOPC
    N = 750
    L = np.sqrt(N * 0.69492)
    nsteps = int(2e6)
    dt = 5e-5
    temp = 310
    nstxout = 10
    nstchk = int(nsteps / 20)
    output = '/Users/eliasnickel/PycharmProjects/BrownLipid/tests/langevin_test/velocity_test'

    base_d_coeff = 0.0562331773557475
    external_forces = {'epsilon': 0.080, 'sigma': 0.60921, 'r_vdw': 1.2, 'r_list': 2.0, 'nstlist': 2,
                       'epsilon_domains': 0.88, 'sigma_domains': 0.7706}

    pressure_coupling = {'ref_p': 0, 'compressibility': 5.6e-05, 'tau_p': 0.01, 'K_A': 240.705,
                         'ref_A': 66.1103/100*N, 'thresh_p': 1e-21, 'nstpcouple': 1}

    mass = 828.0
    langevin_dynamics = {'mass': mass}

    '''uni = brownlipid.Universe(  input_file=False, size_x=L, size_y=L, N=N, nsteps=nsteps, dt=dt,
                                temp=temp, nstxout=nstxout, nstchk=nstchk, base_d_coeff= base_d_coeff,
                                external_forces=external_forces, pressure_coupling=pressure_coupling,
                                langevin_dynamics=langevin_dynamics, output=output)

    uni.evolve()'''

    A = brownlipid.Analysis(input_file='/Users/eliasnickel/Documents/langevin_run/langevin_run_info.json')

    #A.Velocity is loaded in A.get_temperature()
    T, T_var, T_exp_var = A.get_temperature()
    print(f'Relative temperature variance: {T_var:.5g}\nExpected: {T_exp_var:.5g}')

    R = 8.31446

    #Velocity tests
    vel_sq_pp = A.Velocity[:,:,0]**2 + A.Velocity[:,:,1]**2
    vel_rms = np.sqrt(np.mean(vel_sq_pp, axis=1))
    rms_expected = np.sqrt(1e3 * 2 * R * temp / mass)

    vel_sq_mean = np.mean(A.Velocity[:,:,0], axis=1)**2 + np.mean(A.Velocity[:,:,1], axis=1)**2
    vel_rsm = np.sqrt(vel_sq_mean)

    vel_mag = np.sqrt(A.Velocity[:,:,0]**2 + A.Velocity[:,:,1]**2)
    vel_mrs = np.mean(vel_mag, axis=1)

    #Distribution
    vel_flat = A.Velocity.flatten()
    sigma_vel = np.sqrt(R * temp * 1e3 / mass)

    #Plot parameter
    data_skip = 1000 #steps
    fit_start = 0.5 #ns

    # Fits
    x_fit = np.linspace(0, A.nsteps * A.dt, int(A.nsteps / A.nstxout) + 1)[int(fit_start / (A.dt * A.nstxout)):]
    m_T, b_T = np.polyfit(x_fit, T[int(fit_start / (A.dt * A.nstxout)):], 1)
    m_rms, b_rms = np.polyfit(x_fit, vel_rms[int(fit_start / (A.dt * A.nstxout)):], 1)

    #Velocity plot
    x = np.linspace(0, A.nsteps * A.dt, int(A.nsteps / A.nstxout) + 1)[::data_skip]

    plt.figure(figsize=(10,7))
    plt.plot(x, vel_rms[::data_skip], label = 'RMS Velocity' )
    plt.plot(x, np.mean(A.Velocity[:,:,0], axis=1)[::data_skip], label = 'x-Velocity' )
    plt.plot(x, np.mean(A.Velocity[:,:,1], axis=1)[::data_skip], label = 'y-Velocity' )
    #plt.axhline(y=sigma_vel, color='r', linestyle='--', label = 'Theoretical' )
    #plt.plot(x, vel_rsm, label = 'RSM Velocity')
    plt.plot(x, vel_mrs[::data_skip], label = 'MRS Velocity' )
    plt.plot(x_fit, m_rms * x_fit + b_rms, label = f'RMS fit: {m_rms:.2g}x + {b_rms:.2f}\nExpected: {rms_expected:.3g}', color='pink', linewidth=2, linestyle='--')
    plt.xlabel('Time (ns)')
    plt.ylabel('Velocity (nm/ns)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output}_velocity.png')
    plt.show()

    #Distribution plot
    plt.figure(figsize=(10, 7))
    _, bins, _ = plt.hist(vel_flat, bins = 200, density = True, alpha=0.7, label = 'Velocity' )

    theoretical = (1 / (sigma_vel * np.sqrt(2 * np.pi))) * np.exp(-(bins) ** 2 / (2 * sigma_vel ** 2))
    plt.plot(bins, theoretical, label = 'Theoretical' )

    plt.legend()
    plt.xlabel('Velocity (nm/ns)')
    plt.ylabel('Probability Density')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f'{output}_velocity_distr.png')
    plt.show()

    #Temperature plot
    plt.figure(figsize=(10,7))
    plt.plot(x, T[::data_skip], label = 'Temperature' )
    plt.plot(x_fit, m_T * x_fit + b_T, label = f'Fit: {m_T:.2g}x + {b_T:.2f}', color='pink', linewidth=3.5, linestyle='--')
    plt.axhline(y=temp, color='r', linestyle='--', label = f'Input Temp: {A.temp}' ,alpha=0.9)
    plt.xlabel('Time (ns)')
    plt.ylabel('Temperature (K)')
    plt.grid(True)
    plt.legend()
    plt.title(f'Relative temperature variance: {T_var:.5g}, Expected: {T_exp_var:.5g}')
    plt.tight_layout()
    plt.savefig(f'{output}_temperature.png')
    plt.show()


def test_temp_multi_comp():

    input_file = '/Users/eliasnickel/DPPC_DIPC_CHOL_mem/run/20KA_run_info.json'

    A = brownlipid.Analysis(input_file=input_file)

    temp = A.temp
    nsteps = A.nsteps
    dt = A.dt
    nstxout = A.nstxout
    #output = '/Users/eliasnickel/PycharmProjects/BrownLipid/tests/langevin_test/velocity_test_multi_comp'
    output = A.output

    #Temperature
    T, T_var, T_exp_var = A.get_temperature(name=None)

    print(f'Relative temperature variance: {T_var:.5g}\nExpected: {T_exp_var:.5g}')

    #Plot parameter
    data_skip = 1 #steps
    fit_start = 0.5 #ns

    # Fits
    x_fit = np.linspace(0, nsteps * dt, int(nsteps / nstxout) + 1)[int(fit_start / (A.dt * A.nstxout)):]
    m_T, b_T = np.polyfit(x_fit, T[int(fit_start / (A.dt * A.nstxout)):], 1)

    x = np.linspace(0, nsteps * dt, int(nsteps / nstxout) + 1)[::data_skip]

    #Temperature plot
    plt.figure(figsize=(10,7))
    plt.plot(x, T[::data_skip], label = 'Temperature' )
    plt.plot(x_fit, m_T * x_fit + b_T, label = f'Fit: {m_T:.2g}x + {b_T:.2f}', color='pink', linewidth=3.5, linestyle='--')
    plt.axhline(y=temp, color='r', linestyle='--', label = 'Input Temperature' ,alpha=0.9)
    plt.xlabel('Time (ns)')
    plt.ylabel('Temperature (K)')
    plt.grid(True)
    plt.legend()
    plt.title(f'Relative temperature variance: {T_var:.5g}, Expected: {T_exp_var:.5g}')
    plt.tight_layout()
    plt.savefig(f'{output}_temperature.png')
    plt.show()

def test_diffusion():
    N = 100
    L = 8.212134021251742
    nsteps = int(1e8)
    dt = 5e-5
    temp = 310
    nstxout = 25
    nstchk = int(nsteps / 20)
    output = '/Users/eliasnickel/DPPC_DIPC_CHOL_mem/run/20KA'

    base_d_coeff = 0.0562331773557475
    external_forces = {'epsilon': 0.080, 'sigma': 0.60921, 'r_vdw': 1.2, 'r_list': 2.0, 'nstlist': 2,
                       'epsilon_domains': 0.88, 'sigma_domains': 0.7706}

    pressure_coupling = {'ref_p': 0, 'compressibility': 5.6e-05, 'tau_p': 0.01, 'K_A': 240.705,
                         'ref_A': 66.1103 / 100 * N, 'thresh_p': 1e-21, 'nstpcouple': 1}

    mass = 828.0
    langevin_dynamics = {'mass': mass}

    '''uni = brownlipid.Universe(input_file=False, size_x=L, size_y=L, N=N, nsteps=nsteps, dt=dt,
                              temp=temp, nstxout=nstxout, nstchk=nstchk, base_d_coeff=base_d_coeff,
                              external_forces=external_forces, pressure_coupling=pressure_coupling,
                              langevin_dynamics=langevin_dynamics, output=output)

    #uni.evolve()'''

    A = brownlipid.Analysis(input_file=f'{output}_run_info.json')

    fit_start = 25
    fit_stop = 50
    name = 'DPPC'

    #Calc and save
    tau, msd, _ = A.mean_square_displacement(name=name, stop = None)
    np.save(arr=[msd, tau], file=f'{A.output}_msd.npy')

    msd = np.load(f'{A.output}_msd.npy')[0]
    tau = np.load(f'{A.output}_msd.npy')[1]

    d_coeff_fit, intercept = A.mean_square_displacement_fit(tau=tau, msd=msd, dim=2, begin=fit_start, stop=fit_stop)
    np.save(arr=[d_coeff_fit, intercept], file=f'{A.output}_{name}_Dcoeff.npy')

    Dcoeff, intercept = np.load(f'{A.output}_{name}_Dcoeff.npy')

    #Full MSD plot
    plt.figure(figsize=(10, 6))
    plt.plot(tau, msd)
    plt.axvline(fit_start, c='g', ls='--', alpha=0.3, label='fit range')
    plt.axvline(fit_stop, c='g', ls='--', alpha=0.3)
    plt.xlabel('Time (ns)')
    plt.ylabel('MSD (nm$^2$)')
    if name != 'lipid': plt.title(f'MSD ({name})')
    else: plt.title(f'MSD')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{A.output}_{name}_MSD_full.png', dpi=300)
    plt.show()

    #MSD start plot
    plt.figure(figsize=(10, 6))
    plt.plot(tau[:int(200 / A.dt / A.nstxout)], msd[:int(200 / A.dt / A.nstxout)])
    plt.axvline(fit_start, c='g', ls='--', alpha=0.3, label='fit range')
    plt.axvline(fit_stop, c='g', ls='--', alpha=0.3)
    plt.xlabel('Time (ns)')
    plt.ylabel('MSD (nm$^2$)')
    if name != 'lipid': plt.title(f'MSD ({name})')
    else: plt.title(f'MSD')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{A.output}_{name}_MSD_start.png', dpi=300)
    plt.show()

    #MSD fit plot
    start_idx = int(fit_start / tau[-1] * len(tau))
    stop_idx = int(fit_stop / tau[-1] * len(tau))

    m = Dcoeff * 4

    plt.figure(figsize=(10, 6))
    plt.plot(tau, msd)
    plt.plot(tau[start_idx:stop_idx], m * tau[start_idx:stop_idx] + intercept, 'red',
             label=f'Linear fit\nDcoeff={Dcoeff * 1e3:.3g} x10$^{-8}$ cm$^2$/s', ls='--', lw=2.1)
    plt.xlabel('Time (ns)')
    plt.ylabel('MSD (nm$^2$)')
    if name != 'lipid': plt.title(f'MSD ({name})')
    else: plt.title(f'MSD')
    plt.grid(True)
    plt.xlim(fit_start - 12, fit_stop * 1.05)
    plt.ylim(0, np.max(msd[start_idx:stop_idx]) * 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{A.output}_{name}_MSD_fit.png', dpi=300)
    plt.show()

    #A.export_trajectory(pml_file=True, skip = dt * 1000)

def test_rdf_pdf():
    # DOPC
    N = 100
    L = 8.212134021251742
    nsteps = int(1e8)
    dt = 5e-5
    temp = 310
    nstxout = 25
    nstchk = int(nsteps / 20)
    output = '/Users/eliasnickel/Documents/langevin_run/langevin'

    base_d_coeff = 0.0562331773557475
    external_forces = {'epsilon': 0.080, 'sigma': 0.60921, 'r_vdw': 1.2, 'r_list': 2.0, 'nstlist': 2,
                       'epsilon_domains': 0.88, 'sigma_domains': 0.7706}

    pressure_coupling = {'ref_p': 0, 'compressibility': 5.6e-05, 'tau_p': 0.01, 'K_A': 240.705,
                         'ref_A': 66.1103 / 100 * N, 'thresh_p': 1e-21, 'nstpcouple': 1}

    mass = 828.0
    langevin_dynamics = {'mass': mass}

    '''uni = brownlipid.Universe(input_file=False, size_x=L, size_y=L, N=N, nsteps=nsteps, dt=dt,
                              temp=temp, nstxout=nstxout, nstchk=nstchk, base_d_coeff=base_d_coeff,
                              external_forces=external_forces, pressure_coupling=pressure_coupling,
                              langevin_dynamics=langevin_dynamics, output=output)

    #uni.evolve()'''

    A = brownlipid.Analysis(input_file=False, size_x=L, size_y=L, N=N, nsteps=nsteps, dt=dt,
                            temp=temp, nstxout=nstxout, nstchk=nstchk, base_d_coeff=base_d_coeff,
                            external_forces=external_forces, pressure_coupling=pressure_coupling,
                            langevin_dynamics=langevin_dynamics, output=output)

    binwidth = 0.02

    A.load_data_Area()

    #Calc and save
    binmids, rdf, cdf, pdf = A.get_rdf(name_1='lipid', name_2='lipid', r_max=4.0, binwidth=binwidth)
    #np.save(arr=[rdf, binmids], file=f'{A.output}_rdf.npy')
    #np.save(arr=[pdf, binmids], file=f'{A.output}_pdf.npy')

    '''#Load
    rdf = np.load(f'{A.output}_rdf.npy')[0]
    binmids = np.load(f'{A.output}_rdf.npy')[1]
    pdf = np.load(f'{A.output}_pdf.npy')[0]
    pdf /= (np.sum(pdf) * binwidth)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(binmids, rdf, label='RDF')
    axs[1].plot(binmids, pdf, label='Normalized PDF')

    for i in range(2):
        axs[i].set_xlabel('r (nm)')
        axs[i].grid(True)
        axs[i].legend()

    axs[0].set_ylabel('RDF')
    axs[1].set_ylabel('PDF')

    plt.suptitle(f'Radial Distribution and Pair Distance Function')
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{A.output}_plot.png', dpi=300)'''

def test_Area():
    A = brownlipid.Analysis(input_file='/Users/eliasnickel/DPPC_DIPC_CHOL_mem/run/multi_comp_run_info.json')

    A.load_data_Area()

    x = np.linspace(0, A.nsteps * A.dt, int(A.nsteps / A.nstxout) + 1)

    fit_start = 0.66
    fit_index = int((fit_start * len(x)))
    x_fit = x[fit_index:]
    m_A, b_A = np.polyfit(x_fit, A.Area[fit_index:], 1)

    plt.figure(figsize=(8, 6))

    plt.plot(x_fit, A.Area[fit_index:], label='Area', c='C0') #after thresh
    plt.plot(x[:fit_index+1], A.Area[:fit_index+1], alpha=0.5, c='C0') #until thresh
    plt.scatter(x[:fit_index],A.Area[:fit_index],c='b',s=10) #until thresh
    plt.plot(x, np.ones(len(A.Area))*A.ref_A, label=f'ref_A', c='g', linestyle='--', linewidth=2)
    plt.plot(x_fit, m_A*x_fit + b_A, label=f'fit: {m_A:.1g}x + {b_A:.2f}', c='r', linestyle='-.', linewidth=2)
    plt.scatter(-0.001,A.size_x**2,c='purple',s=40,marker='x', alpha=0.7)
    plt.xlabel('Time (ns)')
    plt.ylabel(f'Area (nm$^2$)')
    plt.title(f'Area (K_A: {A.K_A:.4g} mN/m, ref_A: {A.ref_A:.4g} nm$^2$)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{A.output}_Area.png', dpi=300)
    plt.show()