BrownLipid - A library for 2-dimensional lipid diffusion using brownian dynamics
--------------------------------------------------------------------------------

BrownLipid is a Python-based library to perform numerical simulations using brownian dynamics in combination with Van der Waals interactions to study the diffusion of lipids.
The package was originally written for the publication "Bottom-up investigation of spatiotemporal glycocalyx dynamics with interferometric scattering microscopy"

INSTALL
---

We recommend using [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) as virtual enviroment manager for Python:

```
mamba create -n test_brownlipid python=3.13
```

The package versioning is organized using [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

```
curl -sSL https://install.python-poetry.org | python3 -
```

After downloading this repository, install the package from within the repository directory using:

```
poetry install
```

