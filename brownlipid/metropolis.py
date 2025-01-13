import numpy as np
from numba import jit

#Metropolis
@jit(nopython=True)
def metropolis_decision(deltaE, RT):
    
    if deltaE <= 0: return True
    else:
        
        ener = np.array( [ 1.0, np.exp(- deltaE / RT ) ], dtype = np.float32)
        p_accept = np.min( ener )
        alpha    = np.random.rand(1)[0]

        if alpha < p_accept: return True
        else: return False

@jit(nopython=True, fastmath = True)
def fast_metropolis_decision(deltaE):
    alpha = np.random.rand()  # Avoid creating a 1-element array
    return alpha < deltaE  # Simplify the return statement

@jit(nopython=True)
def get_delta_E(x_new, x_old, f_true, N, forceconstant):

    """
    Calculation of the energy difference for changing the number of particles in domains.
    The energy difference is calculated via a simple spring potential.

    """

    #Calculate fraction of particles in domains
    f_old = x_old / N
    f_new = x_new / N

    #Calculate energy levels based on simple spring potential
    E_old = forceconstant * 0.5 * (f_old - f_true)**2 
    E_new = forceconstant * 0.5 * (f_new - f_true)**2

    #Calculate energy difference
    #<= 0 New Energy level is lower or equal -> accept
    # > 0 New Energy level is higher -> prop. reject

    #Unit: kJ/mol
    deltaE = ( E_new - E_old )

    return deltaE
