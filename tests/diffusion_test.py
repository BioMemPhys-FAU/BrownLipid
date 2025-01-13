import brownlipid
import numpy as np
import pytest

@pytest.mark.parametrize("L,d_coeff,N,nsteps,dt", [(100, 1., 1000, 1E5, 1)])
def test_diffusion_coefficient(L,d_coeff,N,nsteps,dt):


    uni = brownlipid.Universe(
                              size_x = L,
                              size_y = L,
                                   N = N,
                             pbc_dim = 'xy',
                              nsteps = int(nsteps),
                                  dt = dt,
                             nstxout = 5,
                        base_d_coeff = d_coeff,
                             nstchk  = int(nsteps)
                            )

    uni.evolve()

    tau, msd, sdpp = uni.mean_square_displacement(skip = 5)

    np.save(arr = tau, file = 'tau_pytest.npy')
    np.save(arr = msd, file = 'msd_pytest.npy')

    d_coeff_fit, intercept = uni.mean_square_displacement_fit(tau = tau, msd = msd, dim = 2, begin = 10., stop = 10000.)

    np.testing.assert_allclose(d_coeff_fit, d_coeff, rtol = 0, atol = 1E-4)
