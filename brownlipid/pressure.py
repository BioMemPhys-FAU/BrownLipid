# ----PYTHON---- #

'''

Berendsen pressure coupling
2D-isotropic barostat

Pressure as tension (mN/m)

dP/dt = - 1/(isothermal_compressibility * A) * dA/dt

scaling_factor = 1 - (compressibility * dt) / (3 * tau_p) * (ref_p - p)
x(t + dt) = scaling_factor * x(t)
size(t + dt) =  scaling_factor * size(t)

'''

import numpy as np

from . import utils
from . import force


def scaling_factor(ref_pos, conf_pos, pairlist, vdw_cutoff, pbc_dim, offsets, A12, B6, N, NkT, area, dt, ref_p, compressibility, tau_p, ext_force_check, thresh_p):


    #Check for external forces and calculate virial
    if ext_force_check != False:

        #Update distances in the self.pairlist
        #New pairlist is created in force.lennard_jones() in forward_in_time() if (frame % nstlist) == 1 or nstlist == 1

        #### BUFFER RADIUS
        #Update pairlist -> Update every distance and distance vector inside the buffer radius of a particle
        pairlist_rij, pairlist_rij_sqrt = force.update_pairlist(ref_pos   = ref_pos, conf_pos  = conf_pos, pairlist  = pairlist, pbc_dim   = pbc_dim, offsets   = offsets[ pairlist[:, 0], pairlist[:, 1] ])

        #### INNER RADIUS
        #Obtain effective distances and distance vectors that are taking into account for the LJ interaction between particles
        _, effective_rij_sqrt, effective_mask, _, _ = force.filter_pairlist(cutoff = vdw_cutoff, rij = pairlist_rij, rij_sq = pairlist_rij_sqrt)

        #Check for pairs in vdw_cutoff
        if effective_rij_sqrt.size == 0:
            virial = np.zeros((N), dtype = np.float32)

        #If there are pairs in vdw_cutoff: calculate virial
        else:
            if effective_rij_sqrt.size > 0: assert effective_rij_sqrt.min() >= 1E-12, f'Too small! {effective_rij_sqrt.min()}'

            masked_pairlist = pairlist[ effective_mask ]

            #Filter A12 and B6 for the effective pairs
            lj_A12 = A12[ masked_pairlist[:, 0], masked_pairlist[:, 1] ]
            lj_B6  = B6[ masked_pairlist[:, 0], masked_pairlist[:, 1] ]

            #Calculate inverse of the squared distance
            inv_rij_sq = 1.0 / effective_rij_sqrt**2

            #Calculate powers for the attractive and the repulsive part of the Lennard-Jones potential
            sr6        = inv_rij_sq ** 3
            sr12       = sr6 ** 2

            sr6        = sr6  *  lj_B6
            sr12       = sr12 * lj_A12

            #Calculate the virial
            virial     = (sr12 - sr6 )

    #No external forces => virial = 0
    else: virial = np.zeros((N), dtype = np.float32)


    #Get pressure of the system
    p = pressure(virial, NkT, area)

    #Check the threshold
    if np.isclose(p, ref_p, atol=thresh_p): return 1, p, virial

    #Calculate scaling factor
    scal_fac = 1 + (compressibility * dt * (p - ref_p) / (3 * tau_p))

    assert scal_fac > 0, f'Scaling factor is invalid! {scal_fac}'

    return scal_fac, p, virial


#Calculate the pressure for scaling
def pressure(virial, NkT, area):

    #[virial] = kJ/mol, but mN*nm needed
    virial_sum = np.sum(virial)  * 6.022E-8

    #Calculate the pressure using the 2D virial equation
    p = NkT / area + (virial_sum / (2 * area))

    #From mN/nm to mN/m:
    return p * 1E9