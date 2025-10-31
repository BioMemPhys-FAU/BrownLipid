import brownlipid
import numpy as np
import pytest

#@pytest.mark.parametrize("L,d_coeff,N,nsteps,dt", [(100, 1., 750, 1E7, 1), (100, 0.5, 750, 1E7, 1.0), (100, np.pi, 750, 5E7, 1.0)])
@pytest.mark.parametrize("L,d_coeff,N,nsteps,dt", [(100, np.pi, 750, 5E7, 1.0)])
def test_diffusion_coefficient(L,d_coeff,N,nsteps,dt):


    uni = brownlipid.Universe(
                              size_x = L,
                              size_y = L,
                                   N = N,
                             pbc_dim = 'xy',
                              nsteps = int(nsteps),
                                  dt = dt,
                             nstxout = 100,
                        base_d_coeff = d_coeff,
                             nstchk  = int(nsteps),
                              output = f"diffusion_test/output_D_{d_coeff:.3f}"
                            )

    uni.evolve()

    tau, msd, sdpp = uni.mean_square_displacement()

    np.save(arr = tau, file = 'diffusion_test/tau_pytest.npy')
    np.save(arr = msd, file = 'diffusion_test/msd_pytest.npy')

    d_coeff_fit, intercept = uni.mean_square_displacement_fit(tau = tau, msd = msd, dim = 2, begin = 0., stop = 100000.)

    print("True Diffusion:", d_coeff)
    print("Sim. Diffusion:", d_coeff_fit)

    np.testing.assert_allclose(d_coeff_fit, d_coeff, rtol = 0, atol = 1E-2)
