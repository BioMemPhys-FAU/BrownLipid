import brownlipid
import numpy as np
import pytest

@pytest.mark.parametrize("L,d_coeff,N,dim,nsteps,dt,grid", [(100, 1., 100000, 2, 1E5, 1, 0.1)])
def test_diffusion_coefficient(L,d_coeff,N,dim,nsteps,dt,grid):


    uni = brownlipid.Universe(
                              size_x = L,
                              size_y = L,
                              size_z = L,
                                   N = N,
                                 dim = dim,
                              nsteps = int(nsteps),
                                  dt = dt,
                             nstxout = 100,
                        base_d_coeff = d_coeff,
                                grid = grid
                            )

    uni.evolve()

    tau, msd, sdpp = uni.mean_square_displacement(skip = dt)

    slope, intercept = np.polyfit(tau, msd, deg = 1)

    d_coeff_fit = slope / 4 / uni.dt

    np.testing.assert_allclose(d_coeff, d_coeff_fit, rtol = 0, atol = 1E-5)
