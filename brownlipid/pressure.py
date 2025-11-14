# ----PYTHON---- #

'''

Berendsen pressure coupling
2D-isotropic barostat

Pressure as tension (mN/m)

dP/dt = - 1/(isothermal_compressibility * A) * dA/dt

linear:         scaling_factor = 1 + (compressibility * dt * (p - ref_p) / (3 * tau_p))
exponential:    scaling_factor = (1 + (compressibility * dt * (p - ref_p) / tau_p)) ** (1/3)

x(t + dt) = scaling_factor * x(t)
size(t + dt) =  scaling_factor * size(t)

'''

import numpy as np

from . import utils
from . import force


def scaling_factor(NkT, area, dt, virial, ref_p, compressibility, tau_p, thresh_p):

    #Get pressure of the system
    p = pressure(virial, NkT, area)

    #Check the threshold
    if np.isclose(p, ref_p, atol=thresh_p): return 1, p

    #Calculate scaling factor
    #scal_fac = 1 + (compressibility * dt * (p - ref_p) / (3 * tau_p))          #linear
    scal_fac = (1 + (compressibility * dt * (p - ref_p) / tau_p)) ** (1 / 3)    #exponential

    assert scal_fac > 0, f'Scaling factor is invalid! {scal_fac}'

    return scal_fac, p


#Calculate the pressure for scaling
def pressure(virial, NkT, area):

    #[virial] = kJ/mol = 6.022E-1 kJ
    virial_sum = np.sum(virial) * 1/6.022

    #Calculate the pressure using the 2D virial equation
    # 1E-23J/nm^2 + 1E-23kJ/nm^2 = 1E-2mN/m + 10mN/m
    p = NkT * 1E-2 / area + virial_sum * 10 / (2 * area)

    return p