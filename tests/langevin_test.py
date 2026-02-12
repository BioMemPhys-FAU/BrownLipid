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
    temp = 310
    R = 8.31446

    for _ in range(20):
        momentum = utils.generate_initial_momenta(N, mass, temp)

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


def test_velocity():
    #DOPC
    '''N = 750
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
    #pressure_coupling=False

    mass = 828.0
    langevin_dynamics = {'mass': mass}'''

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

    '''uni = brownlipid.Universe(  input_file=False, size_x=L, size_y=L, N=N, nsteps=nsteps, dt=dt,
                                temp=temp, nstxout=nstxout, nstchk=nstchk, base_d_coeff= base_d_coeff,
                                external_forces=external_forces, pressure_coupling=pressure_coupling,
                                langevin_dynamics=langevin_dynamics, output=output)

    #uni.evolve()'''

    A = brownlipid.Analysis(input_file=False, size_x=L, size_y=L, N=N, nsteps=nsteps, dt=dt,
                                temp=temp, nstxout=nstxout, nstchk=nstchk, base_d_coeff= base_d_coeff,
                                external_forces=external_forces, pressure_coupling=pressure_coupling,
                                langevin_dynamics=langevin_dynamics, output=output)

    A.load_data_velocity()

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

    #Temperature
    v_sq = np.sum(A.Velocity** 2, axis = (1,2))
    T = ((mass * 1e-3) * v_sq) / (R * (2 * N - 2))

    T_var = np.sqrt(np.var(T, axis = 0)) / np.mean(T)
    T_exp_var = np.sqrt(1/N)
    print(f'Relative temperature variance: {T_var:.5g}\nExpected: {T_exp_var:.5g}')

    #p_sq = np.sum((A.Velocity * mass)**2, axis = (1,2))
    #T = (p_sq / (N*mass)) / (R * (2 * N - 2))

    #Plot parameter
    data_skip = 1000 #steps
    fit_start = 0.5 #ns

    # Fits
    x_fit = np.linspace(0, nsteps * dt, int(nsteps / nstxout) + 1)[int(fit_start / A.dt):]
    m_T, b_T = np.polyfit(x_fit, T[int(fit_start / A.dt):], 1)
    m_rms, b_rms = np.polyfit(x_fit, vel_rms[int(fit_start / A.dt):], 1)

    #Velocity plot
    x = np.linspace(0, nsteps * dt, int(nsteps / nstxout) + 1)[::data_skip]

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

    if pressure_coupling: plt.title('Pressure Coupling enabled')
    else: plt.title('Pressure Coupling disabled')
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
    plt.axhline(y=temp, color='r', linestyle='--', label = 'Input Temperature' ,alpha=0.9)
    plt.xlabel('Time (ns)')
    plt.ylabel('Temperature (K)')
    plt.grid(True)
    plt.legend()
    plt.title(f'Relative temperature variance: {T_var:.5g}, Expected: {T_exp_var:.5g}')
    plt.tight_layout()
    plt.savefig(f'{output}_temperature.png')
    plt.show()

    #A.export_trajectory(pml_file=True, skip = dt * 1000)

def test_small_visual_comparison():
    # DOPC
    N = 10
    L = np.sqrt(N * 0.69492) * 1.2
    nsteps = int(5e5)
    dt = 1e-5
    temp = 310
    nstxout = 1
    nstchk = nsteps

    base_d_coeff = 0.0562331773557475
    external_forces = {'epsilon': 0, 'sigma': 0.60921, 'r_vdw': 1.2, 'r_list': 2.0, 'nstlist': 2,
                       'epsilon_domains': 0.88, 'sigma_domains': 0.7706}

    pressure_coupling = {'ref_p': 0, 'compressibility': 5.6e-05, 'tau_p': 0.01, 'K_A': 240.705,
                         'ref_A': 66.1103 / 100 * N, 'thresh_p': 1e-21, 'nstpcouple': 1}
    pressure_coupling = False

    #Langevin enabled
    mass = 828.0
    langevin_dynamics = {'mass': mass}
    output = '/Users/eliasnickel/PycharmProjects/BrownLipid/tests/langevin_test/lange'

    uni_langevin = brownlipid.Universe(input_file=False, size_x=L, size_y=L, N=N, nsteps=nsteps, dt=dt,
                              temp=temp, nstxout=nstxout, nstchk=nstchk, base_d_coeff=base_d_coeff,
                              external_forces=external_forces, pressure_coupling=pressure_coupling,
                              langevin_dynamics=langevin_dynamics, output=output)

    uni_langevin.evolve()

    A = brownlipid.Analysis(input_file=False, size_x=L, size_y=L, N=N, nsteps=nsteps, dt=dt,
                            temp=temp, nstxout=nstxout, nstchk=nstchk, base_d_coeff=base_d_coeff,
                            external_forces=external_forces, pressure_coupling=pressure_coupling,
                            langevin_dynamics=langevin_dynamics, output=output)

    A.export_trajectory(pml_file=True, skip = dt * 1000)

    #Langevin disabled
    langevin_dynamics = False
    output = '/Users/eliasnickel/PycharmProjects/BrownLipid/tests/langevin_test/no_lange'

    uni_no_langevin = brownlipid.Universe(input_file=False, size_x=L, size_y=L, N=N, nsteps=nsteps, dt=dt,
                                       temp=temp, nstxout=nstxout, nstchk=nstchk, base_d_coeff=base_d_coeff,
                                       external_forces=external_forces, pressure_coupling=pressure_coupling,
                                       langevin_dynamics=langevin_dynamics, output=output)

    uni_no_langevin.evolve()

    A = brownlipid.Analysis(input_file=False, size_x=L, size_y=L, N=N, nsteps=nsteps, dt=dt,
                            temp=temp, nstxout=nstxout, nstchk=nstchk, base_d_coeff=base_d_coeff,
                            external_forces=external_forces, pressure_coupling=pressure_coupling,
                            langevin_dynamics=langevin_dynamics, output=output)

    A.export_trajectory(pml_file=True, skip=dt * 1000)


def test_diffusion():
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

    '''A = brownlipid.Analysis(input_file=False, size_x=L, size_y=L, N=N, nsteps=nsteps, dt=dt,
                            temp=temp, nstxout=nstxout, nstchk=nstchk, base_d_coeff=base_d_coeff,
                            external_forces=external_forces, pressure_coupling=pressure_coupling,
                            langevin_dynamics=langevin_dynamics, output=output)'''

    A = brownlipid.Analysis(input_file='/Users/eliasnickel/Documents/langevin_run/langevin_run_info.json')

    fit_start = 25
    fit_stop = 50

    #Calc and save
    tau, msd, _ = A.mean_square_displacement(stop = None)
    np.save(arr=[msd, tau], file=f'{A.output}_msd.npy')

    msd = np.load(f'{A.output}_msd.npy')[0]
    tau = np.load(f'{A.output}_msd.npy')[1]

    d_coeff_fit, intercept = A.mean_square_displacement_fit(tau=tau, msd=msd, dim=2, begin=fit_start, stop=fit_stop)
    np.save(arr=[d_coeff_fit, intercept], file=f'{A.output}_Dcoeff.npy')

    Dcoeff = np.load(f'{A.output}_Dcoeff.npy')[0]
    intercept = np.load(f'{A.output}_Dcoeff.npy')[1]

    #Full MSD plot
    plt.figure(figsize=(10, 6))
    plt.plot(tau, msd)
    plt.axvline(fit_start, c='g', ls='--', alpha=0.3, label='fit range')
    plt.axvline(fit_stop, c='g', ls='--', alpha=0.3)
    plt.xlabel('Time (ns)')
    plt.ylabel('MSD (nm$^2$)')
    plt.title(f'MSD')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{A.output}_MSD_full.png', dpi=300)
    plt.show()

    #MSD start plot
    plt.figure(figsize=(10, 6))
    plt.plot(tau[:int(200 / dt / nstxout)], msd[:int(200 / dt / nstxout)])
    plt.axvline(fit_start, c='g', ls='--', alpha=0.3, label='fit range')
    plt.axvline(fit_stop, c='g', ls='--', alpha=0.3)
    plt.xlabel('Time (ns)')
    plt.ylabel('MSD (nm$^2$)')
    plt.title(f'MSD')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{A.output}_MSD_start.png', dpi=300)
    plt.show()

    #MSD fit plot
    start_idx = int(fit_start / tau[-1] * len(tau))
    stop_idx = int(fit_stop / tau[-1] * len(tau))

    m = Dcoeff * 4

    plt.figure(figsize=(10, 6))
    plt.plot(tau, msd)
    plt.plot(tau[start_idx:stop_idx], m * tau[start_idx:stop_idx] + intercept, 'red',
             label=f'Linear fit\nDcoeff={Dcoeff * 1e3:.3g} cm$^2$/s', ls='--', lw=2.1)
    plt.xlabel('Time (ns)')
    plt.ylabel('MSD (nm$^2$)')
    plt.title(f'MSD')
    plt.grid(True)
    plt.xlim(fit_start - 12, fit_stop * 1.05)
    plt.ylim(0, np.max(msd[start_idx:stop_idx]) * 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{A.output}_MSD_fit.png', dpi=300)
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

    A.load_data_area()

    exp_density = N / np.mean(A.Area[int(50/dt):])

    #Calc and save
    #binmids, rdf, cdf, pdf = A.get_rdf(exp_density=exp_density, r_max=4.0, binwidth=binwidth)
    #np.save(arr=[rdf, binmids], file=f'{A.output}_rdf.npy')
    #np.save(arr=[pdf, binmids], file=f'{A.output}_pdf.npy')

    #Load
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
    fig.savefig(f'{A.output}_plot.png', dpi=300)



