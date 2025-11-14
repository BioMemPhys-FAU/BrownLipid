from PIL.ImageOps import posterize

import brownlipid
import numpy as np
import matplotlib.pyplot as plt
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
    NkT = N * 1.38E-11 * temp  # In mN * nm
    virial = np.zeros(N)

    p_expected = ((N * 1.38E-23 * temp) / area) * 1E21

    np.testing.assert_allclose(pressure.pressure(virial, NkT, area), p_expected, rtol = 0, atol = 1E-2)

def test_pressure_simple_virial():
    N = 3
    temp = 298
    area = 2
    NkT = N * 1.38E-11 * temp  # In mN * nm
    virial = np.array([1,5,3], dtype = np.float32)

    p_expected = np.float32(((N * 1.38E-23 * temp) / area + 9 * 6.022E-20 / (2 * area)) * 1E21)

    np.testing.assert_allclose(pressure.pressure(virial, NkT, area), p_expected, rtol = 0, atol = 1E-2)


#---------------------------------------------------------------------------------------------------------------------
#Tests for scaling_factor()

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
        #scal_fac = 1 + (compressibility * dt * (p - ref_p) / (3 * tau_p))          #Linear: handles pressure up to ~3.4E4
        scal_fac = (1 + (compressibility * dt * (p - ref_p) / tau_p)) ** (1 / 3)    #Exponential: handles pressure up to ~1.1E4

        assert scal_fac > 0, 'invalid scaling factor'
        assert scal_fac > 1, f'scaling factor <1 although pressure is too high \n p={p}, ref_p={ref_p}, scal_fac={scal_fac}'


        #Pressure too low => scaling factor should be <1

        ref_p = np.random.uniform(low,high)
        p = np.random.uniform(0,ref_p - 0.01)

        #From pressure.py, to be tested:
        #scal_fac = 1 + (compressibility * dt * (p - ref_p) / (3 * tau_p))          #Linear: handles pressure up to ~3.4E4
        scal_fac = (1 + (compressibility * dt * (p - ref_p) / tau_p)) ** (1 / 3)    #Exponential: handles pressure up to ~1.1E4

        assert scal_fac > 0, 'invalid scaling factor'
        assert scal_fac < 1, f'scaling factor >1 although pressure is too low \n p={p}, ref_p={ref_p}, scal_fac={scal_fac}'


def test_pressure_adjustment(
        L=int(10),
        N=130,
        nsteps=int(1E5),
        nstxout=int(500),
        dt=0.2,
        temp=298,
        ref_p=0,
        compressibility=1/300,
        tau_p=10,
        thresh_p = 0.2,
        fit_start = 0.4,        #Fraction of total time
        xlim = [0,1],         #[xl,xr] with Fraction of total time or False
        ylim = True,             #y-scaling in the xlim interval
        virialPlot = False       #either scal_fac or virial plot
        ):

    nstchk = nsteps
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
                                #hard_boundaries={"Circle": [[np.inf, 1.0, 2.5, 2.5]]},
                                #diffusion_domains=0.0,
                                nstchk=nstchk,
                                output='pressure_test/output',
                                external_forces={'epsilon': 0.3221, 'sigma': 0.7706, 'r_vdw': 1.2, 'r_list':3.0, 'nstlist':2, 'epsilon_domains':0.88, 'sigma_domains':0.7706},
                                pressure_coupling={'ref_p': ref_p, 'compressibility': compressibility, 'tau_p': tau_p, 'thresh_p': thresh_p}
                            )

    uni.evolve()

    Pressure = np.load(f'pressure_test/output_Pressure.{1:05d}.npy')[1:]
    Area     = np.load(f'pressure_test/output_Area.{1:05d}.npy')[1:]
    potE     = np.load(f'pressure_test/output_potE.{1:05d}.npy')[1:]
    Virial   = np.load(f'pressure_test/output_Virial.{1:05d}.npy')[1:]
    ScalFac  = np.load(f'pressure_test/output_ScalFac.{1:05d}.npy')[1:]

    #Linear Fitting
    x = np.linspace(0, nsteps * dt, int(nstchk / nstxout))

    fit_index = int((fit_start * len(x)))
    x_fit = x[fit_index:]

    m_p, b_p = np.polyfit(x_fit, Pressure[fit_index:], 1)
    m_A, b_A = np.polyfit(x_fit, Area[fit_index:], 1)

    #RMSD
    Pressure_rmsd = np.sqrt(np.mean((Pressure - ref_p) ** 2))
    Pressure_rmsd_fit = np.sqrt(np.mean((Pressure[fit_index] - ref_p) ** 2))
    print(f'Pressure RMSD: {Pressure_rmsd:.3f} mN/m')
    print(f'Pressure RMSD after fit threshold: {Pressure_rmsd_fit:.3f} mN/m')

    #Plotting
    params = f'L = {L}, N = {N}, ref_p = {ref_p}, compressibility = {compressibility:.3g}, tau_p = {tau_p}, thresh_p = {thresh_p}, nstxout = {nstxout}'

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 8))

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
    axs[0,1].plot(x_fit, m_A*x_fit + b_A, label=f'fit: {m_A:.2g}x + {b_A:.2f}', c='r', linestyle='-.', linewidth=2)
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
        axs[1,1].plot(x, Virial, label='Virial')
        axs[1,1].set_xlabel('Time in ns')
        axs[1,1].set_ylabel('Energy in kJ/mol')
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
    fig.savefig(f'pressure_test/pressure_adjustment.png', dpi=300)
    plt.show()

    np.testing.assert_allclose(b_p, ref_p, atol=1.8)



def test_pos_visual(
        L=int(10),
        N=130,
        nsteps=int(100),
        nstxout=int(1),
        nstchk=int(4),
        dt=0.2,
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
                                #hard_boundaries={"Circle": [[np.inf, 1.0, 2.5, 2.5]]},
                                #diffusion_domains=0.0,
                                nstchk=nstchk,
                                output='pressure_test/output',
                                external_forces={'epsilon': 0.3221, 'sigma': 0.7706, 'r_vdw': 1.2, 'r_list':3.0, 'nstlist':2, 'epsilon_domains':0.88, 'sigma_domains':0.7706},
                                pressure_coupling={'ref_p': ref_p, 'compressibility': compressibility, 'tau_p': tau_p}
                            )

    uni.evolve()

    Pressure = np.load(f'pressure_test/output_Pressure.{1:05d}.npy')[1:]
    Area     = np.load(f'pressure_test/output_Area.{1:05d}.npy')
    Area[0]  = L**2
    potE     = np.load(f'pressure_test/output_potE.{1:05d}.npy')
    Virial   = np.load(f'pressure_test/output_Virial.{1:05d}.npy')
    ScalFac  = np.load(f'pressure_test/output_ScalFac.{1:05d}.npy')

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