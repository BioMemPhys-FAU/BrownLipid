BrownLipid - A library for 2-dimensional lipid diffusion using brownian dynamics
--------------------------------------------------------------------------------

BrownLipid is a Python-based library to perform numerical simulations using brownian dynamics in combination with Van der Waals (VdW) interactions to study the diffusion of lipids.
The package was originally written for the publication "Bottom-up investigation of spatiotemporal glycocalyx dynamics with interferometric scattering microscopy"

INSTALL
---

We recommend using [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) as virtual enviroment manager for Python:

```bash
mamba create -n test_brownlipid python=3.13
```

The package versioning is organized using [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

After downloading this repository, install the package from within the repository directory using:

```bash
poetry install
```

DISCLAIMER
---

The package is still under construction (e.g., the algorithms for diffusion under hydrodynamics don't work yet). Version 0.1.0 was used for the above mentioned publication.

USAGE
---

The following code snippet shows the setup of a simple simulation using BrownLipid.

```python
import brownlipid
import numpy as np

#Set simulation parameters
uni = brownlipid.Universe(
                          #Lateral length (nm)
                          size_x            = 5,
                          size_y            = 5,
                          #Number of lipids in the system
                          N                 = 50,
                          #Periodic boundary conditions
                          pbc_dim           = 'xy',
                          #Number of simulation steps
                          nsteps            = 10000,
                          #Time step in ns
                          dt                = 0.2,
                          #Place a circle with radius 1.0 nm into the plane center (The np.inf is ignored)
                          hard_boundaries   = {"Circle":[[np.inf, 1.0, 2.5, 2.5]]},
                          #The circle is fixed
                          diffusion_domains = 0.0,
                          #The circle interacts via VdW with the lipids
                          softwall          = True,
                          #Output frequency (steps)
                          nstxout           = 100,
                          #Save output every nstchk steps to disk
                          nstchk  = int(1.0E3),
                          #Output directory
                          output  = f"output/simple",
                          #Diffusion coefficent of the lipids (nm^2 / ns)
                          base_d_coeff      = 257.799593E-6,
                          #LJ interaction parameters
                          external_forces   = {'epsilon': 0.3221, 'sigma': 0.7706,               #Interaction between lipids
                                               'epsilon_domains': 0.88, 'sigma_domains': 0.7706, #Interaction between domains
                                               'r_vdw': 1.2, 'r_list':3.0,                       #VdW cutoff and buffer cutoff
                                               'nstlist':10}                                     #Update pairlist every 10 steps
                          )

#Run simulation
uni.evolve()
```


