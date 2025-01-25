import brownlipid
import numpy as np
import pytest

@pytest.mark.parametrize("L,d_coeff,N,nsteps,dt", [(100, 1., 500, 1E7, 1), (100, 0.5, 500, 1E7, 1.0)])
def test_diffusion_coefficient(L,d_coeff,N,nsteps,dt):

    rep_diff = []
    for i in range(1):

        uni = brownlipid.Universe(
                                  size_x = L,
                                  size_y = L,
                                       N = N,
                                 pbc_dim = 'xy',
                                  nsteps = int(nsteps),
                                      dt = dt,
                                 nstxout = 100,
                            base_d_coeff = d_coeff,
                                 nstchk  = int(nsteps)
                                )

        uni.evolve()

        tau, msd, sdpp = uni.mean_square_displacement()

        np.save(arr = tau, file = 'tau_pytest.npy')
        np.save(arr = msd, file = 'msd_pytest.npy')

        d_coeff_fit, intercept = uni.mean_square_displacement_fit(tau = tau, msd = msd, dim = 2, begin = 0., stop = 100000.)

        rep_diff.append( d_coeff_fit )

    np.testing.assert_allclose(np.mean(rep_diff), d_coeff, rtol = 0, atol = 1E-3)
