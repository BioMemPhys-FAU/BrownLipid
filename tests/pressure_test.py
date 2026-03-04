from PIL.ImageOps import posterize

import brownlipid
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pytest

from brownlipid import pressure
from brownlipid import utils
from brownlipid import force

#---------------------------------------------------------------------------------------------------------------------
#Tests for pressure()

def test_pressure_zero_virial():
    N = 4
    temp = 298
    area = 2
    ref_A = area
    K_A = 1
    NkT = N * 1.38 * temp  # In mN * nm
    virial = np.zeros(N)
    p_lrc_const = 0

    p_expected = ((N * 1.38E-23 * temp) / area) * 1E21

    np.testing.assert_allclose(pressure.pressure(virial, NkT, area, K_A ,ref_A, p_lrc_const), p_expected, rtol = 0, atol = 1E-2)

def test_pressure_simple_virial():
    N = 3
    temp = 298
    area = 2
    ref_A = area
    K_A = 1
    NkT = N * 1.38 * temp #1E-23 J
    virial = np.array([1,5,3], dtype = np.float32)
    p_lrc_const = 0

    p_expected = np.float32((N * 1.38E-2 * temp) / area + (9 / 6.022) * 10 / (2 * area))

    np.testing.assert_allclose(pressure.pressure(virial, NkT, area, K_A, ref_A, p_lrc_const), p_expected, rtol = 0, atol = 1E-2)

def test_pressure_simple_virial_with_correction():
    N = 3
    temp = 298
    area = 2
    ref_A = area
    K_A = 1
    NkT = N * 1.38 * temp #1E-23 J
    virial = np.array([1,5,3], dtype = np.float32)
    p_lrc_const = -0.2

    p_expected = np.float32((N * 1.38E-2 * temp) / area + (9 / 6.022) * 10 / (2 * area))
    p_expected += - 1/6.022 * 10 * 0.2 / 4

    np.testing.assert_allclose(pressure.pressure(virial, NkT, area, K_A, ref_A,  p_lrc_const), p_expected, rtol = 0, atol = 1E-2)


#---------------------------------------------------------------------------------------------------------------------
#Test for scaling_factor()

def test_scaling_calc():

    compressibility = 1 / 230
    dt = 1.0
    tau_p = 50

    low = 0.1
    high = 20

    for _ in range(int(1E5)):

        #Pressure too high => scaling factor should be >1

        ref_p = np.random.uniform(low,high)
        p = np.random.uniform(ref_p + 0.01, high)

        #From pressure.py, to be tested:
        #scal_fac = 1 + (compressibility * dt * (p - ref_p) / (2 * tau_p))          #Linear equation can handle higher pressure values
        scal_fac = (1 + (compressibility * dt * (p - ref_p) / tau_p)) ** (1 / 2)

        assert scal_fac > 0, 'invalid scaling factor'
        assert scal_fac > 1, f'scaling factor <1 although pressure is too high \n p={p}, ref_p={ref_p}, scal_fac={scal_fac}'


        #Pressure too low => scaling factor should be <1

        ref_p = np.random.uniform(low,high)
        p = np.random.uniform(0,ref_p - 0.01)

        #From pressure.py, to be tested:
        #scal_fac = 1 + (compressibility * dt * (p - ref_p) / (2 * tau_p))          #Linear equation can handle higher pressure values
        scal_fac = (1 + (compressibility * dt * (p - ref_p) / tau_p)) ** (1 / 2)

        assert scal_fac > 0, 'invalid scaling factor'
        assert scal_fac < 1, f'scaling factor >1 although pressure is too low \n p={p}, ref_p={ref_p}, scal_fac={scal_fac}'


#---------------------------------------------------------------------------------------------------------------------
#Tests for pressure coupling

def test_pressure_adjustment_run(
        #input = '/Volumes/lacie_1/elias/cg_mem_run/analysis/parameter_files/DPPC_whole_lipid_params.json',
        input = '/Users/eliasnickel/DPPC_DIPC_CHOL_mem/run/multi_component_run_info.json',

        #DOPC
        N = 100,
        L = np.sqrt((66.1103/100)*100),
        comp_name = ['DOPC'],
        nsteps = int(1e3),
        dt = 5e-5,
        temp = 310,
        nstxout = 20,
        output = 'pressure_test/output',

        base_d_coeff = 0.0562331773557475,
        external_forces = {'epsilon': 0.080, 'sigma': 0.60921, 'r_vdw': 1.2, 'r_list': 2.0, 'nstlist': 2,
                           'epsilon_domains': 0.88, 'sigma_domains': 0.7706},

        ref_p = 0,
        compressibility = 5.6e-05,
        tau_p = 0.01,
        K_A = 240.705,
        ref_A = (66.1103/100)*100,
        thresh_p = 1e-21,
        nstpcouple = 1,

        langevin_dynamics={'mass': 828},

        fit_start = 0.37,       #fraction of total time
        test_tol = 1.8,
        xlim = [],              #[xl,xr] with Fraction of total time or False
        ylim = True,            #y-scaling in the xlim interval
        virialPlot = True,      #either scal_fac or virial plot
        correlation = False,
        t_equ = 20             #for correlation
        ):

    nstchk = nsteps

    uni = brownlipid.Universe(  input_file=False,
                                size_x=L,
                                size_y=L,
                                N=N,
                                comp_name=comp_name,
                                nsteps=nsteps,
                                dt=dt,
                                temp=temp,
                                nstxout=nstxout,
                                base_d_coeff= base_d_coeff,
                                nstchk=nstchk,
                                output=output,
                                external_forces=external_forces,
                                pressure_coupling={'ref_p': ref_p, 'compressibility': compressibility, 'tau_p': tau_p, 'K_A': K_A, 'ref_A': ref_A, 'thresh_p': thresh_p, 'nstpcouple': nstpcouple},
                                langevin_dynamics=langevin_dynamics
                              )

    uni.evolve()

    assert nsteps % nstchk == 0, 'nsteps must be divisible by nstchk'

    if correlation: assert t_equ < dt * nsteps, 't_equ must be smaller than the total simulation time'

    '''p_lrc_const = 12 * N**2 * np.pi * external_forces['epsilon'] * external_forces['sigma']**2 * (0.2 * (external_forces['sigma'] / external_forces['r_vdw'])**10 - 0.25 * (external_forces['sigma'] / external_forces['r_vdw'])**4)
    print('Constant part of long-range correction for pressure:', p_lrc_const / 6.022, '* 1e-35 mN * m^3')'''

    '''Area_expected = (2**(1/6) * external_forces['sigma'] / 2)**2 * np.pi * N
    L_expected = np.sqrt(Area_expected)
    virial_expected = 2*(ref_p*Area_expected * 6.022*0.1 - N * 1.38 * temp * 6.022*1e-3 - 12 * N**2 / Area_expected * 3.14159 * external_forces['epsilon'] * external_forces['sigma']**2 * (0.2 * (external_forces['sigma']/external_forces['r_vdw'])**10 - 0.25 * (external_forces['sigma']/external_forces['r_vdw'])**4))
    print(f'Expected area: {Area_expected:.3g} nm^2\nExpected box length: {L_expected:.3g} nm\nExpected virial: {virial_expected:.3g} kJ/mol')'''

    Pressure = np.load(f'{output}_Pressure.{1:05d}.npy')[1:]
    Area     = np.load(f'{output}_Area.{1:05d}.npy')[1:]
    potE     = np.load(f'{output}_potE.{1:05d}.npy')[1:]
    Virial   = np.load(f'{output}_Virial.{1:05d}.npy')[1:]
    ScalFac  = np.load(f'{output}_ScalFac.{1:05d}.npy')[1:]

    #Linear Fitting
    x = np.linspace(0, nsteps * dt, int(nstchk / nstxout))

    fit_index = int((fit_start * len(x)))
    x_fit = x[fit_index:]

    actual_virial = sum(Virial[fit_index:]) / len(Virial[fit_index:])
    print(f'Actual virial (mean): {sum(Virial[fit_index:]) / len(Virial[fit_index:]):.3f} kJ/mol')
    m_p, b_p = np.polyfit(x_fit, Pressure[fit_index:], 1)
    m_A, b_A = np.polyfit(x_fit, Area[fit_index:], 1)

    #RMSD
    Pressure_rmsd = np.sqrt(np.mean((Pressure - ref_p) ** 2))
    Pressure_rmsd_fit = np.sqrt(np.mean((Pressure[fit_index] - ref_p) ** 2))
    print(f'Pressure RMSD: {Pressure_rmsd:.3f} mN/m')
    print(f'Pressure RMSD after fit threshold: {Pressure_rmsd_fit:.3f} mN/m')

    #Plotting
    #params = (f'L = {L:.3g}, N = {N}, ref_p = {ref_p}, eps = {external_forces['epsilon']}, sig = {external_forces['sigma']}, r_vdw = {external_forces['r_vdw']}, D = {base_d_coeff:.3g}\n'
    #          f'compressibility = {compressibility:.3g}, tau_p = {tau_p:.3g}, K_A = {K_A:.3g}, ref_A = {ref_A:.3g}, thresh_p = {thresh_p}, nstpcouple =  {nstpcouple:.3g}, dt = {dt:.3g}, nstxout = {nstxout:.3g}\n')
              #f'Expected area: {Area_expected:.3g} nm^2, expected box length: {L_expected:.3g} nm, expected virial: {virial_expected:.3g} kJ/mol, actual virial (mean): {actual_virial:.3f} kJ/mol')

    params = (f'L = {L:.3g}, N = {np.sum(N)}, ref_p = {ref_p}, r_vdw = {external_forces['r_vdw']},\n'
              f'compressibility = {compressibility:.3g}, tau_p = {tau_p:.3g}, K_A = {K_A:.3g}, ref_A = {ref_A:.3g}, thresh_p = {thresh_p}, nstpcouple =  {nstpcouple:.3g}, dt = {dt:.3g}, nstxout = {nstxout:.3g}\n')


    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 8.5))

    #Pressure plot
    axs[0,0].plot(x_fit, Pressure[fit_index:], label='Pressure',alpha=0.9 ,c='C0')       #after thresh
    axs[0,0].plot(x[:fit_index+1], Pressure[:fit_index+1], alpha=0.5, c='C0')  #until thresh
    axs[0,0].scatter(x[:fit_index],Pressure[:fit_index],c='b',s=10)                 #until thresh
    axs[0,0].plot(x, np.ones(len(Pressure))*ref_p, label=f'ref_p (RMSD: {Pressure_rmsd:.2f})', c='g', linestyle='--', linewidth=2)
    axs[0,0].plot(x_fit, m_p*x_fit + b_p, label=f'fit: {m_p:.2g}x + {b_p:.2f}', c='r',alpha = 0.8, linestyle='-.', linewidth=2)
    axs[0,0].set_xlabel('Time in ns')
    axs[0,0].set_ylabel('Pressure in mN/m')
    axs[0,0].grid(True)
    axs[0,0].legend()

    #Area plot
    axs[0,1].plot(x_fit, Area[fit_index:], label='Area', c='C0') #after thresh
    axs[0,1].plot(x[:fit_index+1], Area[:fit_index+1], alpha=0.5, c='C0') #until thresh
    axs[0,1].scatter(x[:fit_index],Area[:fit_index],c='b',s=10) #until thresh
    axs[0,1].plot(x, np.ones(len(Area))*ref_A, label=f'ref_A', c='g', linestyle='--', linewidth=2)
    axs[0,1].plot(x_fit, m_A*x_fit + b_A, label=f'fit: {m_A:.2g}x + {b_A:.2f}', c='r', linestyle='-.', linewidth=2)
    axs[0,1].scatter(-0.001,L**2,c='purple',s=40,marker='x', alpha=0.7)
    axs[0,1].set_xlabel('Time in ns')
    axs[0,1].set_ylabel('Area in nm^2')
    axs[0,1].grid(True)
    axs[0,1].legend()

    #Energy plot
    axs[1,0].plot(x, potE, label='pot. Energy')
    axs[1,0].set_xlabel('Time in ns')
    axs[1,0].set_ylabel('Energy in kJ/mol')
    axs[1,0].grid(True)
    axs[1,0].legend()

    if virialPlot:
        #Virial plot
        axs[1,1].plot(x, Virial, label='Virial (not LR-corrected)')
        axs[1,1].set_xlabel('Time in ns')
        axs[1,1].set_ylabel('Virial in kJ/mol')
        axs[1,1].grid(True)
        axs[1,1].legend()

    else:
        #Scaling Factor plot
        axs[1, 1].plot(x_fit, ScalFac[fit_index:], label='Scaling Factor', alpha=0.9, c='C0')  # after thresh
        axs[1, 1].plot(x[:fit_index + 1], ScalFac[:fit_index + 1], alpha=0.5, c='C0')  # until thresh
        axs[1, 1].scatter(x[:fit_index], ScalFac[:fit_index], c='b', s=10) # until thresh
        axs[1, 1].set_xlabel('Time in ns')
        axs[1, 1].set_ylabel('Scaling Factor')
        axs[1, 1].grid(True)
        axs[1, 1].legend()

    #set plot limits
    if xlim:
        xlcut = nsteps * dt * xlim[0]
        xrcut = nsteps * dt * xlim[1]
        axs[0, 0].set_xlim(xlcut, xrcut)
        axs[0, 1].set_xlim(xlcut, xrcut)
        axs[1, 0].set_xlim(xlcut, xrcut)
        axs[1, 1].set_xlim(xlcut, xrcut)
        if ylim:
            xli = int(xlcut / dt / nstxout)
            xri = int(xrcut / dt / nstxout)
            axs[0, 0].set_ylim(min(Pressure[xli:xri]), max(Pressure[xli:xri]))
            axs[0, 1].set_ylim(min(Area[xli:xri]), max(Area[xli:xri]))
            axs[1, 0].set_ylim(min(potE[xli:xri]), max(potE[xli:xri]))
            if virialPlot:
                axs[1, 1].set_ylim(min(Virial[xli:xri]), max(Virial[xli:xri]))
            else:
                axs[1, 1].set_ylim(min(ScalFac[xli:xri]), max(ScalFac[xli:xri]))

    plt.suptitle(f'Parameters: {params}', fontsize=13)
    plt.tight_layout()
    fig.savefig(f'{output}_pressure_adjustment.png', dpi=300)
    plt.show()


    #Correlation
    if correlation:
        assert t_equ < dt * nsteps, 't_equ must be smaller than the total simulation time'

        Pressure = np.load(f'pressure_test/output_Pressure.{1:05d}.npy')[int(t_equ // dt // nstxout) + 1:]
        Area = np.load(f'pressure_test/output_Area.{1:05d}.npy')[int(t_equ // dt // nstxout) + 1:]
        n_p = len(Pressure)

        #Get the centered arrays
        mean_p = np.mean(Pressure)
        Pressure -= mean_p
        mean_a = np.mean(Area)
        Area -= mean_a

        #Calculate correlation, normalize it and extract for positive lags
        cor_pp = np.correlate(Pressure,Pressure,mode='full')
        if min(cor_pp) == 0 and max(cor_pp) == 0: cor_pp_filt = np.zeros(len(cor_pp[n_p - 1:]))
        else:
            cor_pp_norm = cor_pp / cor_pp[n_p - 1]
            cor_pp_filt = cor_pp_norm[n_p - 1:]

        cor_pa = np.correlate(Pressure,Area,mode='full')
        if min(cor_pa) == 0 and max(cor_pa) == 0: cor_pa_filt = np.zeros(len(cor_pa[n_p - 1:]))
        else: cor_pa_filt = cor_pa[n_p - 1:] / np.sqrt(np.sum(Pressure**2) * np.sum(Area**2))

        #Plotting
        Pressure = np.load(f'pressure_test/output_Pressure.{1:05d}.npy')[1:]
        Area = np.load(f'pressure_test/output_Area.{1:05d}.npy')[1:]
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 8), dpi=300)
        tau = np.linspace(0, (n_p - 1) * dt * nstxout, n_p)
        t = np.linspace(0, nsteps * dt, int(nstchk / nstxout))

        axs[0,0].scatter(tau,cor_pp_filt, s=5, c='C0', label='Pressure Autocorrelation')
        axs[0,0].plot(tau,cor_pp_filt, c='orange', alpha=0.35)
        axs[0,0].set_xlabel('Time lag in ns')
        axs[0,0].set_ylabel('Normalized Autocorrelation')
        axs[0,0].grid(True)

        axs[1,0].plot(t, Pressure, label='Pressure')
        axs[1,0].set_xlabel('Time in ns')
        axs[1,0].set_ylabel('Pressure in mN/m')
        axs[1,0].grid(True)

        axs[0,1].scatter(tau,cor_pa_filt, s=5, c='C0', label='Pressure Area Correlation')
        axs[0,1].plot(tau,cor_pa_filt, c='orange', alpha=0.35)
        axs[0,1].set_xlabel('Time lag in ns')
        axs[0,1].set_ylabel('Normalized Crosscorrelation')
        axs[0,1].grid(True)

        axs[1,1].plot(t, Area, label='Area')
        axs[1,1].set_xlabel('Time in ns')
        axs[1,1].set_ylabel('Area in nm^2')
        axs[1,1].grid(True)

        #Fit A(tau)=a*e**(tau/tau _c)+c
        def simpleExp(tau, tau_c, a, c):
            return a * np.exp(-tau / tau_c) + c

        params, cv = scipy.optimize.curve_fit(f=simpleExp, xdata=tau, ydata=cor_pp_filt, p0=[10*dt*nstxout,1,0])
        tau_c, a, c = params
        axs[0,0].plot(tau, simpleExp(tau, tau_c, a, c), c='r', label=f'Fit: A(tau) = {a:.2f}*e^(-tau/{tau_c:.2f})+{c:.2f}', ls='--')

        axs[1,0].plot(np.ones(2)*(tau_c+t_equ), np.linspace(min(Pressure), max(Pressure),2), alpha=0.8, label='tau_c', ls='-.')
        axs[1,0].plot(np.ones(2)*t_equ, np.linspace(min(Pressure), max(Pressure),2), alpha=0.8, label='t_equ', ls='--')
        axs[1,1].plot(np.ones(2)*t_equ, np.linspace(min(Area), max(Area),2), alpha=0.8, c ='g', label='t_equ', ls='--')

        #Set plot limits
        if xlim:
            xlcut = nsteps * dt * xlim[0]
            xrcut = nsteps * dt * xlim[1]
            xlcut_c = (nsteps * dt - t_equ) * xlim[0]
            xrcut_c = (nsteps * dt - t_equ) * xlim[1]
            axs[0, 0].set_xlim(xlcut_c, xrcut_c)
            axs[0, 1].set_xlim(xlcut_c, xrcut_c)
            axs[1, 0].set_xlim(xlcut, xrcut)
            axs[1, 1].set_xlim(xlcut, xrcut)
            if ylim:
                xli = int(xlcut / dt / nstxout)
                xri = int(xrcut / dt / nstxout)
                xli_c = int(xlim[0] * n_p)
                xri_c = int(xlim[1] * n_p)
                axs[1,0].set_ylim(min(Pressure[xli:xri]), max(Pressure[xli:xri]))
                axs[1,1].set_ylim(min(Area[xli:xri]), max(Area[xli:xri]))
                axs[0,0].set_ylim(min(cor_pp_filt[xli_c:xri_c]), max(cor_pp_filt[xli_c:xri_c]))
                axs[0,1].set_ylim(min(cor_pa_filt[xli_c:xri_c]), max(cor_pa_filt[xli_c:xri_c]))

        params = f'L = {L}, N = {N}, ref_p = {ref_p}, eps = {external_forces['epsilon']}, sig = {external_forces['sigma']}, compressibility = {compressibility:.3g}, tau_p = {tau_p:.3g}, thresh_p = {thresh_p},\nnstpcouple =  {nstpcouple:.3g}, dt = {dt:.3g}, nstxout = {nstxout:.3g}, tau_equ = {t_equ}'
        plt.suptitle(f'{params}', fontsize=13)
        axs[0,0].legend()
        axs[0,1].legend()
        axs[1,0].legend()
        axs[1,1].legend()
        plt.tight_layout()
        fig.savefig(f'{output}_pressure_adjustment_correlation.png', dpi=300)
        plt.show()

        print('tau_c = ', tau_c, 'ns')

    #np.testing.assert_allclose(b_p, ref_p, atol=test_tol)
    # return actual_virial

def test_pressure_adjustment_file(
        input_file='/Users/eliasnickel/DPPC_DIPC_CHOL_mem/run/multi_comp_run_info.json',
        fit_start = 0.37,       #fraction of total time
        xlim = [],              #[xl,xr] with Fraction of total time or False
        ylim = True,            #y-scaling in the xlim interval
        virialPlot = True,      #either scal_fac or virial plot
        ):

    A = brownlipid.Analysis(input_file=input_file)

    A.load_data_Pressure()
    A.load_data_Area()
    start = 100 / A.nstxout
    kBT = 1.38 * A.temp
    A_msf = np.mean((A.Area[int(start / A.dt):] - np.mean(A.Area[int(start / A.dt):]))**2)
    K_A = 1e-2 * kBT * np.mean(A.Area[int(start / A.dt):]) / A_msf
    print('K_A:', K_A)
    A.load_data_potEnergy()
    A.load_data_Virial()
    A.load_data_ScalFac()

    L = A.size_x

    #Linear Fitting
    x = np.linspace(0, A.nsteps * A.dt, int(A.nsteps / A.nstxout) + 1)

    fit_index = int((fit_start * len(x)))
    x_fit = x[fit_index:]

    m_p, b_p = np.polyfit(x_fit, A.Pressure[fit_index:], 1)
    m_A, b_A = np.polyfit(x_fit, A.Area[fit_index:], 1)

    #RMSD
    Pressure_rmsd = np.sqrt(np.mean((A.Pressure - A.ref_p) ** 2))
    Pressure_rmsd_fit = np.sqrt(np.mean((A.Pressure[fit_index] - A.ref_p) ** 2))
    print(f'Pressure RMSD: {Pressure_rmsd:.3f} mN/m')
    print(f'Pressure RMSD after fit threshold: {Pressure_rmsd_fit:.3f} mN/m')

    #Plotting
    params = (f'L = {L:.3g}, N = {A.N}, ref_p = {A.ref_p}, r_vdw = {A.external_forces['r_vdw']}, nstxout = {A.nstxout:.3g}\n'
              f'compressibility = {A.compressibility:.3g}, tau_p = {A.tau_p:.3g}, K_A = {A.K_A:.3g}, ref_A = {A.ref_A:.3g}, '
              f'thresh_p = {A.thresh_p}, nstpcouple =  {A.nstpcouple:.3g}, dt = {A.dt:.3g}')

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 8.5))

    #Pressure plot
    axs[0,0].plot(x_fit, A.Pressure[fit_index:], label='Pressure',alpha=0.9 ,c='C0')       #after thresh
    axs[0,0].plot(x[:fit_index+1], A.Pressure[:fit_index+1], alpha=0.5, c='C0')  #until thresh
    axs[0,0].scatter(x[:fit_index],A.Pressure[:fit_index],c='b',s=10)                 #until thresh
    axs[0,0].plot(x, np.ones(len(A.Pressure))*A.ref_p, label=f'ref_p (RMSD: {Pressure_rmsd:.2f})', c='g', linestyle='--', linewidth=2)
    axs[0,0].plot(x_fit, m_p*x_fit + b_p, label=f'fit: {m_p:.2g}x + {b_p:.2f}', c='r',alpha = 0.8, linestyle='-.', linewidth=2)
    axs[0,0].set_xlabel('Time in ns')
    axs[0,0].set_ylabel('Pressure in mN/m')
    axs[0,0].grid(True)
    axs[0,0].legend()

    #Area plot
    axs[0,1].plot(x_fit, A.Area[fit_index:], label='Area', c='C0') #after thresh
    axs[0,1].plot(x[:fit_index+1], A.Area[:fit_index+1], alpha=0.5, c='C0') #until thresh
    axs[0,1].scatter(x[:fit_index],A.Area[:fit_index],c='b',s=10) #until thresh
    axs[0,1].plot(x, np.ones(len(A.Area))*A.ref_A, label=f'ref_A', c='g', linestyle='--', linewidth=2)
    axs[0,1].plot(x_fit, m_A*x_fit + b_A, label=f'fit: {m_A:.2g}x + {b_A:.2f}', c='r', linestyle='-.', linewidth=2)
    axs[0,1].scatter(-0.001,L**2,c='purple',s=40,marker='x', alpha=0.7)
    axs[0,1].set_xlabel('Time in ns')
    axs[0,1].set_ylabel('Area in nm^2')
    axs[0,1].grid(True)
    axs[0,1].legend()

    #Energy plot
    axs[1,0].plot(x, A.potEnergy, label='pot. Energy')
    axs[1,0].set_xlabel('Time in ns')
    axs[1,0].set_ylabel('Energy in kJ/mol')
    axs[1,0].grid(True)
    axs[1,0].legend()

    if virialPlot:
        #Virial plot
        axs[1,1].plot(x, A.Virial, label='Virial (not LR-corrected)')
        axs[1,1].set_xlabel('Time in ns')
        axs[1,1].set_ylabel('Virial in kJ/mol')
        axs[1,1].grid(True)
        axs[1,1].legend()

    else:
        #Scaling Factor plot
        axs[1, 1].plot(x_fit, A.ScalFac[fit_index:], label='Scaling Factor', alpha=0.9, c='C0')  # after thresh
        axs[1, 1].plot(x[:fit_index + 1], A.ScalFac[:fit_index + 1], alpha=0.5, c='C0')  # until thresh
        axs[1, 1].scatter(x[:fit_index], A.ScalFac[:fit_index], c='b', s=10) # until thresh
        axs[1, 1].set_xlabel('Time in ns')
        axs[1, 1].set_ylabel('Scaling Factor')
        axs[1, 1].grid(True)
        axs[1, 1].legend()

    #set plot limits
    if xlim:
        xlcut = A.nsteps * A.dt * xlim[0]
        xrcut = A.nsteps * A.dt * xlim[1]
        axs[0, 0].set_xlim(xlcut, xrcut)
        axs[0, 1].set_xlim(xlcut, xrcut)
        axs[1, 0].set_xlim(xlcut, xrcut)
        axs[1, 1].set_xlim(xlcut, xrcut)
        if ylim:
            xli = int(xlcut / A.dt / A.nstxout)
            xri = int(xrcut / A.dt / A.nstxout)
            axs[0, 0].set_ylim(min(A.Pressure[xli:xri]), max(A.Pressure[xli:xri]))
            axs[0, 1].set_ylim(min(A.Area[xli:xri]), max(A.Area[xli:xri]))
            axs[1, 0].set_ylim(min(A.potEnergy[xli:xri]), max(A.potEnergy[xli:xri]))
            if virialPlot:
                axs[1, 1].set_ylim(min(A.Virial[xli:xri]), max(A.Virial[xli:xri]))
            else:
                axs[1, 1].set_ylim(min(A.ScalFac[xli:xri]), max(A.ScalFac[xli:xri]))

    plt.suptitle(f'Parameters: {params}', fontsize=13)
    plt.tight_layout()
    fig.savefig(f'{A.output}_pressure_adjustment.png', dpi=300)
    plt.show()

    A.export_trajectory(pml_file=True, skip=1)

#---------------------------------------------------------------------------------------------------------------------


def test_pos_visual(
        L=int(18),
        N=130,
        nsteps=int(100),
        nstxout=int(1),
        nstchk=int(4),
        dt=1e-2,
        temp=298,
        ref_p=0,
        compressibility=1/300,
        tau_p=10,
        pressure_plot = True
        ):

    assert nsteps % nstchk == 0, 'nsteps must be divisible by nstchk'

    uni = brownlipid.Universe(
                                size_x=L,
                                size_y=L,
                                N=N,
                                nsteps=nsteps,
                                dt=dt,
                                temp=temp,
                                nstxout=nstxout,
                                base_d_coeff=248.916E-6,
                                nstchk=nstchk,
                                output='pressure_test/output',
                                external_forces={'epsilon': 0.3221, 'sigma': 0.7706, 'r_vdw': 99, 'r_list':100, 'nstlist':2, 'epsilon_domains':0.88, 'sigma_domains':0.7706},
                                pressure_coupling={'ref_p': ref_p, 'compressibility': compressibility, 'tau_p': tau_p}
                            )

    uni.evolve()

    Pressure = np.load(f'pressure_test/output_Pressure.{1:05d}.npy')[1:]
    Area     = np.load(f'pressure_test/output_Area.{1:05d}.npy')
    Area[0]  = L**2

    Pos = np.load(f'pressure_test/output_wrap.{1:05d}.npy')
    xpos = Pos[:,:,0]
    ypos = Pos[:,:,1]

    # Plotting
    params = f'L = {L}, N = {N}, ref_p = {ref_p}, compressibility = {compressibility:.3g}, tau_p = {tau_p}'

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 8))

    t = np.linspace(dt, nsteps * dt, int(nstchk / nstxout))

    axs[0,0].scatter(xpos[0], ypos[0], c='C0', s=10, label=f't=0')
    axs[0,0].legend()
    axs[0,1].scatter(xpos[1], ypos[1], c='C1', s=10)
    axs[1,0].scatter(xpos[2], ypos[2], c='C2', s=10)
    if not pressure_plot: axs[1,1].scatter(xpos[3], ypos[3], c='C3', s=10)

    for i in range(2):
        for j in range(2):

            axs[i,j].set_xlabel('x in nm')
            axs[i,j].set_ylabel('y in nm')

    if pressure_plot:
        axs[1,1].plot(t, Pressure, label='Pressure')
        axs[1,1].set_xlabel('Time in ns')
        axs[1,1].set_ylabel('Pressure in mN/m')
        axs[1,1].grid(True)
        axs[1,1].legend()

    plt.suptitle(f'Parameters: {params}', fontsize=13)
    plt.tight_layout()
    plt.show()

    print(f'Pressure at start: {Pressure[0]:.8f} mN/m')


def test_correlation(
        L=int(18),
        N=130,
        nsteps=int(5E4),
        nstxout=int(10),
        dt=5e-5,
        temp=298,
        ref_p=0,
        compressibility=4.5e-5,
        tau_p=1E2,
        t_equ = 2,
        xlim = [],         #[xl,xr] with Fraction of total time or False
        ylim = True
        ):

    nstchk = nsteps

    uni = brownlipid.Universe(
        size_x=L,
        size_y=L,
        N=N,
        nsteps=nsteps,
        dt=dt,
        temp=temp,
        nstxout=nstxout,
        base_d_coeff=248.916E-6,
        nstchk=nstchk,
        output='pressure_test/output',
        external_forces={'epsilon': 0.3221, 'sigma': 0.7706, 'r_vdw': 1.2, 'r_list':3.0, 'nstlist':2, 'epsilon_domains':0.88, 'sigma_domains':0.7706},
        pressure_coupling={'ref_p': ref_p, 'compressibility': compressibility, 'tau_p': tau_p}
    )

    uni.evolve()

    Pressure = np.load(f'pressure_test/output_Pressure.{1:05d}.npy')[int(t_equ // dt // nstxout) + 1:]
    Area = np.load(f'pressure_test/output_Area.{1:05d}.npy')[int(t_equ // dt // nstxout) + 1:]
    n_p = len(Pressure)

    #Get the centered arrays
    mean_p = np.mean(Pressure)
    Pressure -= mean_p
    mean_a = np.mean(Area)
    Area -= mean_a

    #Calculate correlation, normalize it and extract for positive lags
    cor_pp = np.correlate(Pressure,Pressure,mode='full')
    if min(cor_pp) == 0 and max(cor_pp) == 0: cor_pp_filt = np.zeros(len(cor_pp[n_p - 1:]))
    else:
        cor_pp_norm = cor_pp / cor_pp[n_p - 1]
        cor_pp_filt = cor_pp_norm[n_p - 1:]

    cor_pa = np.correlate(Pressure,Area,mode='full')
    if min(cor_pa) == 0 and max(cor_pa) == 0: cor_pa_filt = np.zeros(len(cor_pa[n_p - 1:]))
    else: cor_pa_filt = cor_pa[n_p - 1:] / np.sqrt(np.sum(Pressure**2) * np.sum(Area**2))

    #Plotting
    Pressure = np.load(f'pressure_test/output_Pressure.{1:05d}.npy')[1:]
    Area = np.load(f'pressure_test/output_Area.{1:05d}.npy')[1:]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 8), dpi=300)
    tau = np.linspace(0, (n_p - 1) * dt * nstxout, n_p)
    t = np.linspace(0, nsteps * dt, int(nstchk / nstxout))

    axs[0,0].scatter(tau,cor_pp_filt, s=5, c='C0', label='Pressure Autocorrelation')
    axs[0,0].plot(tau,cor_pp_filt, c='orange', alpha=0.35)
    axs[0,0].set_xlabel('Time lag in ns')
    axs[0,0].set_ylabel('Normalized Autocorrelation')
    axs[0,0].grid(True)

    axs[1,0].plot(t, Pressure, label='Pressure')
    axs[1,0].set_xlabel('Time in ns')
    axs[1,0].set_ylabel('Pressure in mN/m')
    axs[1,0].grid(True)

    axs[0,1].scatter(tau,cor_pa_filt, s=5, c='C0', label='Pressure Area Correlation')
    axs[0,1].plot(tau,cor_pa_filt, c='orange', alpha=0.35)
    axs[0,1].set_xlabel('Time lag in ns')
    axs[0,1].set_ylabel('Normalized Crosscorrelation')
    axs[0,1].grid(True)

    axs[1,1].plot(t, Area, label='Area')
    axs[1,1].set_xlabel('Time in ns')
    axs[1,1].set_ylabel('Area in nm^2')
    axs[1,1].grid(True)

    #Fit A(tau)=a*e**(tau/tau _c)+c
    def simpleExp(tau, tau_c, a, c):
        return a * np.exp(-tau / tau_c) + c

    params, cv = scipy.optimize.curve_fit(f=simpleExp, xdata=tau, ydata=cor_pp_filt, p0=[10*dt*nstxout,1,0])
    tau_c, a, c = params
    axs[0,0].plot(tau, simpleExp(tau, tau_c, a, c), c='r', label=f'Fit: A(tau) = {a:.2f}*e^(-tau/{tau_c:.2f})+{c:.2f}', ls='--')

    axs[1,0].plot(np.ones(2)*(tau_c+t_equ), np.linspace(min(Pressure), max(Pressure),2), alpha=0.8, label='tau_c', ls='-.')
    axs[1,0].plot(np.ones(2)*t_equ, np.linspace(min(Pressure), max(Pressure),2), alpha=0.8, label='t_equ', ls='--')
    axs[1,1].plot(np.ones(2)*t_equ, np.linspace(min(Area), max(Area),2), alpha=0.8, c ='g', label='t_equ', ls='--')

    #Set plot limits
    if xlim:
        xlcut = nsteps * dt * xlim[0]
        xrcut = nsteps * dt * xlim[1]
        xlcut_c = (nsteps * dt - t_equ) * xlim[0]
        xrcut_c = (nsteps * dt - t_equ) * xlim[1]
        axs[0, 0].set_xlim(xlcut_c, xrcut_c)
        axs[0, 1].set_xlim(xlcut_c, xrcut_c)
        axs[1, 0].set_xlim(xlcut, xrcut)
        axs[1, 1].set_xlim(xlcut, xrcut)
        if ylim:
            xli = int(xlcut / dt / nstxout)
            xri = int(xrcut / dt / nstxout)
            xli_c = int(xlim[0] * n_p)
            xri_c = int(xlim[1] * n_p)
            axs[1,0].set_ylim(min(Pressure[xli:xri]), max(Pressure[xli:xri]))
            axs[1,1].set_ylim(min(Area[xli:xri]), max(Area[xli:xri]))
            axs[0,0].set_ylim(min(cor_pp_filt[xli_c:xri_c]), max(cor_pp_filt[xli_c:xri_c]))
            axs[0,1].set_ylim(min(cor_pa_filt[xli_c:xri_c]), max(cor_pa_filt[xli_c:xri_c]))

    params = f'L = {L}, N = {N}, ref_p = {ref_p}, compressibility = {compressibility:.3g}, tau_p = {tau_p:.3g}, tau_equ = {t_equ}, dt = {dt:.3g}, nstxout = {nstxout}'
    plt.suptitle(f'{params}', fontsize=13)
    axs[0,0].legend()
    axs[0,1].legend()
    axs[1,0].legend()
    axs[1,1].legend()
    plt.tight_layout()
    plt.show()

    print('tau_c = ', tau_c, 'ns')