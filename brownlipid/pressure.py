# ----PYTHON---- #

'''

Berendsen pressure coupling
2D-isotropic barostat

Pressure as tension (mN/m)

dP/dt = - 1/(isothermal_compressibility * A) * dA/dt

caling_factor = (1 + (compressibility * dt * (p - ref_p) / tau_p)) ** (1/2)

x(t + dt) = scaling_factor * x(t)
size(t + dt) =  scaling_factor * size(t)

'''

import numpy as np

from . import utils
from . import force


def scaling_factor(NkT, area, dt, virial, p_lrc_const, ref_p, compressibility, tau_p, thresh_p):

    #Get pressure of the system
    p = pressure(virial, NkT, area, p_lrc_const)

    #Check the threshold
    if np.isclose(p, ref_p, atol=thresh_p): return 1, p

    #Calculate scaling factor
    scal_fac = (1 + (compressibility * dt * (p - ref_p) / tau_p)) ** (1 / 2)    #exponential

    assert scal_fac > 0, f'Scaling factor is invalid! {scal_fac}'

    return scal_fac, p


#Calculate the pressure for scaling
def pressure(virial, NkT, area, p_lrc_const):

    #[virial] = kJ/mol = 1/6.022 kJ
    virial_sum = np.sum(virial) * 1/6.022

    #Calculate the pressure using the 2D virial equation
    # 1E-23J/nm^2 + 1E-23kJ/nm^2 = 1E-2mN/m + 10mN/m
    p = NkT * 1E-2 / area + virial_sum * 10 / (2 * area)

    #Calculate and apply 2D long-range correction
    p_lrc = 1/6.022 * 10 * p_lrc_const / area**2
    p += p_lrc

    return p