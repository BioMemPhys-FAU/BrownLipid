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

    low = 0.5
    high = 50

    for _ in range(int(1E5)):
        #Pressure too high => scaling factor should be >1
        compressibility = 1/230
        dt = 1.0
        ref_p = np.random.uniform(low,high)
        p = np.random.uniform(ref_p + 0.01, high)
        tau_p = 20

        #From pressure.py, to be tested:
        scal_fac = 1 + (compressibility * dt * (p - ref_p) / (3 * tau_p))

        assert scal_fac > 0, 'invalid scaling factor'
        assert scal_fac > 1, f'scaling factor <1 although pressure is too high \n p={scal_fac}, ref_p={ref_p}, scal_fac={scal_fac}'

        #Pressure too low => scaling factor should be <1
        compressibility = 1/230
        dt = 1.0
        ref_p = np.random.uniform(low,high)
        p = np.random.uniform(0,ref_p - 0.01)
        tau_p = 20

        # From pressure.py, to be tested:
        scal_fac = 1 + (compressibility * dt * (p - ref_p) / (3 * tau_p))

        assert scal_fac > 0, 'invalid scaling factor'
        assert scal_fac < 1, f'scaling factor >1 although pressure is too high \n p={scal_fac}, ref_p={ref_p}, scal_fac={scal_fac}'


def test_pressure_adjustment(
        L=10,   #Area explodes when L is small
        N=50,
        nsteps=int(1E5),
        nstxout=int(100),
        dt=1,
        temp=298,
        ref_p=7,
        compressibility=1/230,
        tau_p=50,
        fit_start = 0.15    #Fraction of time
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
                                nstchk=nstchk,
                                output='pressure_test/output',
                                external_forces={'epsilon': 0.3221, 'sigma': 0.7706, 'r_vdw': 1.2, 'r_list':3.0, 'nstlist':2, 'epsilon_domains':0.88, 'sigma_domains':0.7706},
                                pressure_coupling={'ref_p': ref_p, 'compressibility': compressibility, 'tau_p': tau_p}
                            )

    uni.evolve()

    Pressure = np.load(f'pressure_test/output_Pressure.{1:05d}.npy')[1:]
    Area     = np.load(f'pressure_test/output_Area.{1:05d}.npy')[1:]

    # RMSD
    Pressure_rmsd = np.sqrt(np.mean((Pressure - ref_p) ** 2))
    print(f'Pressure RMSD: {Pressure_rmsd:.3f} mN/m')

    #Linear Fitting
    x = np.linspace(0, nsteps * dt, int(nstchk / nstxout))

    fit_index = int((fit_start * len(x)))
    x_fit = x[fit_index:]

    m_p, b_p = np.polyfit(x_fit, Pressure[fit_index:], 1)
    m_A, b_A = np.polyfit(x_fit, Area[fit_index:], 1)

    #Plotting
    params = f'L = {L}, N = {N}, ref_p = {ref_p}, compressibility = {compressibility:.3f}, tau_p = {tau_p}'

    plt.figure(dpi=200)
    plt.plot(x, Pressure, label='pressure')
    plt.plot(x, np.ones(len(Pressure))*ref_p, label=f'ref_p (RMSD: {Pressure_rmsd:.3f})', c='r', linestyle='--')
    plt.plot(x_fit, m_p*x_fit + b_p, label=f'fit: {m_p:.5f}x + {b_p:.3f}', c='g',alpha = 0.8, linestyle='-.')
    plt.xlabel('Time in ns')
    plt.ylabel('Pressure in mN/m')
    plt.grid(True)
    plt.text(nsteps*dt*0.94,max(Pressure)*1.12,s=params,fontsize=9,bbox=dict(facecolor='white', alpha=0.5), horizontalalignment='right', verticalalignment='top')
    plt.legend()
    plt.show()

    plt.figure(dpi=200)
    plt.plot(x, Area, label='Area')
    plt.plot(x_fit, m_A*x_fit + b_A, label=f'fit: {m_A:.5f}x + {b_A:.3f}', c='magenta', linestyle='-.')
    plt.xlabel('Time in ns')
    plt.ylabel('Area in nm^2')
    plt.grid(True)
    plt.text(nsteps*dt*0.94,max(Area)*1.1,s=params,fontsize=9,bbox=dict(facecolor='white', alpha=0.5), horizontalalignment='right', verticalalignment='top')
    plt.legend()
    plt.show()

    np.testing.assert_allclose(b_p, ref_p, atol=0.9)